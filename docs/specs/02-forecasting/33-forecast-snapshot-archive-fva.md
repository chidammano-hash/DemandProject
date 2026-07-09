# 33 - Forecast Snapshot Archive & Live FVA

**Status:** Implemented
**Date:** 2026-07-09
**Related:** 03-backtest-framework.md, 08-production-forecast.md, 24-candidate-forecast-promotion.md, 32-lag-decomposed-accuracy-leaderboard.md, ../08-integration/07-fva.md, ../07-user-experience/06-backtest-cleanup.md, ../01-foundation/06-execution-lag.md

---

## 1. Problem

Every month the platform publishes a champion-routed plan from `fact_production_forecast_staging` to `fact_production_forecast`. Staging can hold many generated models, but the archive is intentionally bounded: it retains the promoted champion and exactly three non-champion contenders for each record month.
Neither live table preserves history:

- The next promote executes `DELETE FROM fact_production_forecast` (backtest_management.py promote, step 4), destroying the previously published plan.
- Staging is regenerated per model with DELETE-before-INSERT, so each generation overwrites the last.
- Staging is transient work space for every generated model; retaining every model and its whole horizon would create an unnecessarily large, low-value history.

Consequently the platform can never answer "what did the published champion and its three best alternatives predict for June 2026, as of June 2026, and how accurate were they once actuals landed?"
The existing `backtest_lag_archive` does not cover this: it holds backtest *simulations* (retrospective predictions from simulated training cutoffs), is capped at lag 0-4, and is keyed on the `(item_id, customer_group, loc)` backtest grain.
Production forecasts are real forward statements at the `(item_id, loc)` DFU grain and must not be mixed into backtest accuracy surfaces.

The concrete June 2026 case: forward forecasts generated as-of June 2026 (`forecast_month_generated = 2026-06-01`; promoted champion `plan_version = '2026-06'`) cover June through November 2026. The archive keeps only the champion and the frozen top-three contender roster for those six months.
Those six months are the complete archive scope: snapshot lags 0 through 5 relative to a June record month.
Without an archive, the moment the July cycle regenerates staging and re-promotes, the June predictions are gone and real (non-backtest) accuracy can never be computed.

## 2. Design principles

1. **Snapshot what was believed, keyed by when it was believed.**
   Every archived row carries `record_month` (the planning cycle that produced it) and a derived `lag` (months from record month to forecast month).
2. **Keep a fixed, auditable four-series roster.**
   Every record month retains exactly `champion` plus contender ranks 1, 2, and 3. The contender roster is selected and persisted before its forecasts are generated; archive and FVA reads never re-rank it. All other staged models, including `ai_champion`, are out of scope for this archive.
3. **Plan-age lag and model horizon are different numbers; keep both.**
   Snapshot `lag` is the age of the plan of record ("as of June, what did we publish for August" = lag 2).
   `horizon_months` is the model's true forecast distance from its own per-DFU last-actual origin.
   On the live June generation, 22.6% of series (48,233 of 213,578) have a first forecast month *before* June because their history ends early, so a June-dated row can be true horizon 1 for one DFU and horizon 9 for another.
   The archive therefore carries `horizon_months` verbatim so the two are never conflated, and snapshot lag curves must not be presented as equivalent to the backtest natural-lag curves of `agg_accuracy_lag_archive` (see 6.4).
4. **Archive and cleanup are separate workflows.**
   Archiving is a non-destructive monthly snapshot.
   Cleanup is an independently triggered purge, gated on verifying the archive exists and reconciles, so an operator can archive without committing to a purge and can never delete an un-archived generation.
5. **Real accuracy, not simulated, and honestly labeled.**
   Snapshot accuracy joins archived forward forecasts to actuals as they land at DFU grain. Each contender-versus-champion comparison is recomputed on its common DFU set, because the selected models can have different coverage.
6. **Reuse the platform rails.**
   JobManager job types, `get_planning_date()` (never `date.today()`), `refresh_for_tables()` with the MV registered in `MV_SOURCES`, config in YAML, psycopg `%s` parameters.

## 3. Data model

### 3.1 `forecast_snapshot_roster` (NEW, sql/202)

One immutable model-selection record per archived series and record month. It establishes which three non-champion models are retained before their forecasts are generated; it is not a forecast fact table.

| Column | Type | Description |
|---|---|---|
| record_month | DATE NOT NULL | As-of planning month (month-start CHECK) |
| model_id | VARCHAR(100) NOT NULL | `champion` or a selected non-champion model id |
| snapshot_role | TEXT NOT NULL | `champion` or `contender` |
| contender_rank | SMALLINT | `1`, `2`, or `3` for contenders; NULL for `champion` |
| source_backtest_run_id | INTEGER | FK to `backtest_run.id` that supplied the contender's frozen score; NULL for `champion` |
| rank_wape | NUMERIC | Frozen aggregate backtest WAPE used to rank a contender; NULL for `champion` |
| generation_run_id | UUID | The contender-generation run; NULL for `champion` |
| selected_at | TIMESTAMPTZ NOT NULL DEFAULT NOW() | When the roster was frozen |

- Primary key: `(record_month, model_id)`.
- `snapshot_role = 'champion'` requires `model_id = 'champion'` and a NULL `contender_rank`; a partial unique index permits one champion row per record month.
- `snapshot_role = 'contender'` requires `model_id <> 'champion'`, `contender_rank BETWEEN 1 AND 3`, and a non-NULL `generation_run_id`; a partial unique index permits one contender at each rank per record month.
- The roster-preparation workflow must write exactly one champion row and all three contender ranks in one transaction. Missing, duplicate, or changed ranks abort archive; there is no partial four-series snapshot.
- A contender is chosen from forecastable non-champion models by its latest successfully completed, database-loaded `backtest_run` available when the roster is prepared. Models with NULL WAPE or no completed run are ineligible. Rank by `wape ASC`, `accuracy_pct DESC`, `completed_at DESC`, then `model_id ASC`; persist the chosen run id and WAPE. This makes "top 3" deterministic and prevents later backtest results from rewriting historical selection.

### 3.2 `fact_forecast_snapshot` (NEW, sql/202)

One row per algorithm's prediction for one DFU-month, as of one record month.

| Column | Type | Description |
|---|---|---|
| snapshot_sk | BIGSERIAL PK | Surrogate key |
| record_month | DATE NOT NULL | As-of planning month (month-start CHECK); the user-facing "recordMonth" |
| model_id | VARCHAR(100) NOT NULL | `champion` for the promoted plan or a selected roster contender |
| item_id | VARCHAR(50) NOT NULL | DFU item |
| loc | VARCHAR(50) NOT NULL | DFU location |
| forecast_month | DATE NOT NULL | Target month (month-start CHECK) |
| lag | SMALLINT generated, stored | Plan age in months; see formula below; CHECK `lag BETWEEN 0 AND 5` |
| horizon_months | SMALLINT | The model's true horizon from its per-DFU origin, carried from the source row (NULL for sources that lack it) |
| forecast_qty | NUMERIC(12,2) NOT NULL | Point forecast (P50) |
| forecast_qty_lower | NUMERIC(12,2) | P10 |
| forecast_qty_upper | NUMERIC(12,2) | P90 |
| source_model_id | VARCHAR(100) | For `champion` rows: the routed underlying model |
| cluster_id | TEXT | Cluster at generation time |
| plan_version | VARCHAR(30) | For `champion` rows: the promotion's plan_version |
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
  A lag/date mismatch is impossible by construction. The table also has `CHECK (lag BETWEEN 0 AND 5)`, so no other forecast horizon can enter the archive even if a caller supplies a wider source window.
- Unique key: `(record_month, model_id, item_id, loc, forecast_month)`.
- Foreign key: `(record_month, model_id)` references `forecast_snapshot_roster`. This prevents any model outside the frozen champion-plus-three roster from being archived.
- **Snapshot immutability:** the archive INSERT uses `ON CONFLICT DO NOTHING` - the first snapshot for a record month wins.
  Rationale: the planning date can be pinned (`config/planning_config.yaml`) while the wall clock advances, so a late re-generation can be produced *after* the record month's actuals have loaded; letting it overwrite the earlier snapshot would inject hindsight into "as-of" accuracy.
  A deliberate re-snapshot requires `--overwrite`, which is logged loudly and records the replaced rows' count.
- Indexes: the unique key; `(record_month, lag)`; `(model_id, record_month)`; `(item_id, loc, forecast_month)`.
- Grain note: production forecasts are `(item_id, loc)` DFU grain with **no customer_group**.
  Never join this table to `dim_sku` on only `(item_id, loc)` - dim_sku's grain is 3-key and a 2-key join fans out across customer groups (see 01-accuracy-kpis.md).
  Item-level attributes may be joined via `dim_item` safely.

### 3.3 `agg_accuracy_snapshot` (NEW MV, same sql/202 file)

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
- The MV stores the refresh statement timestamp as `last_refresh_at` on each materialized row; `/fva/snapshot-months` returns its maximum for the selected record month (NULL when no lag is closed yet).
- Unique index on `(record_month, model_id, item_id, loc, forecast_month)` backs `REFRESH ... CONCURRENTLY`.
- Registered in `common/core/mv_refresh.py` `MV_SOURCES` as `agg_accuracy_snapshot: {fact_forecast_snapshot, fact_sales_monthly}` - **Tier 1** (aggregates directly over base tables, like `agg_dfu_naive_scale`), in the same change as the DDL; enforced by `tests/unit/test_mv_refresh.py`.
  Sales loads and archive runs then refresh it automatically via `refresh_for_tables()`.
- Sizing: at most four archived models per `(record_month, lag)`. The actual row count can be lower when a selected contender has narrower DFU coverage.

## 4. Workflow 1 - archive (`archive_forecast_snapshot`)

New script `scripts/forecasting/archive_forecast_snapshot.py` (modern template per scripts/db/refresh_mvs.py: `logging`, module-run form, int exit codes, no `print()`, no module-level `parents[N]`).

### 4.1 Freeze the contender roster and generate its forecasts

New script/job `prepare_forecast_snapshot_contenders` runs after production training and before the next archive deadline. It:

1. Resolves and inserts the immutable roster from the completed, loaded `backtest_run` scores described in 3.1.
2. Allocates one UUID per contender and invokes `generate_production_forecasts.py --model-id <id> --horizon 6 --run-id <uuid>` once for each contender rank. The generator gains the `--run-id` option and writes that exact UUID to staging; the contender roster records it.
3. Fails the publish workflow if any selected contender cannot generate all six archive months. It does not promote, archive, or generate any fourth contender.

The normal champion-generation job remains responsible for the operational plan. The contender job is solely the minimal comparison set needed for live-forward FVA.

### 4.2 Sources and predicates

| Source | Predicate | Yields |
|---|---|---|
| `fact_production_forecast_staging` | Join `forecast_snapshot_roster` where `snapshot_role = 'contender'` **and** `staging.run_id = roster.generation_run_id`; `forecast_month_generated = record_month AND forecast_month >= record_month AND forecast_month < record_month + INTERVAL '6 months'` | Only contender ranks 1-3 from their frozen generation runs, lags 0..5, including `horizon_months` |
| `fact_production_forecast` | Join the roster's `champion` row; `model_id = 'champion' AND plan_version = to_char(record_month, 'YYYY-MM')`, with the same fixed six-month window | The promoted plan with `source_model_id` routing |

- `record_month` defaults to `MAX(forecast_month_generated)` present in staging, validated `<= get_planning_date()` month; `--record-month YYYY-MM` overrides.
  Deriving the record month from the data (not the wall clock) makes a July run correctly archive the June generation.
  **Empty staging** (post-cleanup, pre-generation window): the script logs "nothing to archive" and exits 0 - a scheduled run in that window is a documented no-op, not an error.
- Archive scope is fixed to `record_month` through `record_month + 5 months`. The archive has no `--horizon` or `--models` override.
- **Prerequisite fix (same change):** `promote_model()` currently stamps `plan_version = datetime.now(UTC).strftime("%Y-%m")` (backtest_management.py:998) - wall clock - while generation stamps `forecast_month_generated` from `get_planning_date()`.
  Whenever the planning date is pinned to a different calendar month than the wall clock, a re-promote can write the wall-clock month and silently break the champion predicate.
  Change promote to derive plan_version from `get_planning_date()` so both stamps come from the same clock, per the platform rule.
- A missing or incomplete roster, an absent champion, missing champion or contender rows for any lag 0..5, or a champion whose `plan_version` does not match the record month is a hard failure (exit 2). The script writes no partial snapshot; staging remains available for remediation.
- Inserts are single `INSERT INTO ... SELECT` statements with `ON CONFLICT ... DO NOTHING` (first snapshot wins; `--overwrite` switches to `DO UPDATE`, see 3.2).
- After commit: `refresh_for_tables(["fact_forecast_snapshot"])`, then a per-model archived-row summary in the job log.
- `--dry-run` prints counts for exactly the four roster models. `--overwrite` may replace forecast rows using the same frozen roster; it cannot select a new contender set.

### 4.3 Job, scheduling, and the archive-before-overwrite invariant

- Register JobTypeDef `prepare_forecast_snapshot_contenders` (group `forecast`, subprocess style) and `archive_forecast_snapshot` (params `{record_month: None, dry_run: False, overwrite: False}`).
- **The hard deadline:** a generation's candidates die at the next generation's DELETE-before-INSERT, and the champion dies at the next promote's `DELETE FROM fact_production_forecast`.
  A calendar cron alone can miss that window, losing exactly the history this feature exists to keep.
  Therefore the `forecast-publish` named pipeline (config/forecasting/pipelines.yaml) gains a **leading** `archive_forecast_snapshot` step: the outgoing generation is snapshotted before train/generate replace it. After champion generation, it runs `prepare_forecast_snapshot_contenders` to freeze and generate the next cycle's three alternatives.
  With `ON CONFLICT DO NOTHING` this step is an idempotent no-op when the month-close run already archived, and a no-op on empty staging.
- Belt-and-suspenders cron: `default_schedules` entry `forecast_snapshot_monthly` (cron `0 4 3 * *`, the 3rd at 04:00, after month-open actuals loads; `enabled: false` initially).
- Manual trigger: Jobs tab, or `make forecast-archive ARGS="--record-month 2026-06"`.
- Named bundle: `forecast-snapshot-bundle` runs contender selection, archive, and
  reconciliation-gated cleanup in that order. Its cleanup step targets generations
  older than the planning month; use the standalone cleanup job for an explicit
  `--generation` override.

### 4.4 Sizing (June 2026 reference population)

For the cited 12,306-DFU June population, the upper bound is 295,344 archived rows per record month (4 series x 12,306 DFUs x 6 lags). That is a 78% reduction from retaining the prior every-candidate-plus-champion example (1,355,304 rows); actual volume is lower if a selected contender has narrower coverage.
Coverage asymmetry is why contender-versus-champion FVA deltas use a common-DFU intersection (6.2). Single `INSERT ... SELECT` statements at this scale are comfortably within APScheduler limits; pg-queue is not needed.

## 5. Workflow 2 - cleanup (`cleanup_forecast_staging`)

Separate script `scripts/forecasting/cleanup_forecast_staging.py`, separate job type, never chained automatically after the archive.

### 5.1 Scope

Deletes from `fact_production_forecast_staging` only.
`fact_production_forecast` is not touched: the promoted plan must remain live for planning consumers (inventory, ai_champion, UI), and the next promote already replaces it.

Default target: generations strictly older than the current planning month (`forecast_month_generated < date_trunc('month', get_planning_date())`).
`--generation YYYY-MM` targets one generation explicitly, including the current generation, but it always remains subject to the same non-bypassable archive-first gate.

### 5.2 Safety gate (the archive-first invariant)

Before deleting a generation G, require a complete roster (one `champion`, contender ranks 1-3) and reconcile only the selected contender generations:

```
archived(record_month = G, contender model_id, run_id) =
  staging rows(G, same contender model_id, same frozen run_id, lag 0..5)

AND archived(record_month = G, model_id = 'champion') > 0
```

Any missing roster member or reconciliation shortfall aborts with exit 2 and a per-model reconciliation report. There is no force override for this archive-first safety gate.
Unselected staged models and rows beyond lag 5 are deliberately excluded from this gate and may be deleted without archive compensation. They are regenerated every cycle and are outside the fixed champion-plus-three retention policy.

- `--dry-run` reports what would be deleted and the gate verdict without deleting.
- No MV refresh is needed: no registered MV reads staging (verified against `MV_SOURCES`).
- Register JobTypeDef `cleanup_forecast_staging` in the same `forecast` group: group FIFO guarantees archive and cleanup never run concurrently.
- Make target `forecast-staging-clean ARGS=...`.

## 6. Accuracy & FVA

### 6.1 What becomes measurable

With record month 2026-06 archived and June actuals loaded, lag-0 accuracy for the champion and three frozen contenders is computable immediately.
Each subsequent month's actuals close one more lag: July actuals close lag 1, ..., November actuals close lag 5.
When July 2026 is archived (record month 2026-07), the lag matrix starts filling diagonally; after six cycles every lag has a measured value per model per record month - real forward accuracy alongside (not merged with) the simulated backtest curves.

### 6.2 API

Extend `api/routers/forecasting/fva.py` with cached, replica-tolerant reads: both endpoints use `@cached_sync(...)` with cache keys covering all query parameters and `get_read_only_conn()`. The panel already exposes MV refresh time, so brief replica/cache lag is acceptable and visible.

| Method | Path | Description |
|---|---|---|
| GET | `/fva/snapshot-accuracy?record_month=&lag=` | Required `record_month`; optional `lag` constrained to 0..5. Champion-plus-three accuracy aggregated from `agg_accuracy_snapshot`: `{model_id, snapshot_role, contender_rank, lag, forecast_month, n_dfus, accuracy_pct, wape, bias, fva_vs_champion_pts, n_dfus_common}` rows; omit `lag` for the full model x lag matrix of one record month |
| GET | `/fva/snapshot-months` | Distinct record months available, per-month closed-lag count, and the MV's last-refresh time (drives the UI selector and the freshness caption) |

- `accuracy_pct`, `wape`, `bias` per model are aggregated over that model's own covered DFUs (with `n_dfus` always returned).
- `fva_vs_champion_pts` is computed **only on the DFU intersection** between each contender and `champion` at the same `(record_month, lag)`; both sides are re-aggregated over that common set (`n_dfus_common`). A positive value means the contender outperformed the promoted plan on the same DFUs. The champion's value is `0`; an empty intersection is `NULL`/n-a.
  A whole-universe delta would be coverage bias, not forecast skill.
- This bounded live-forward panel does not require `seasonal_naive` to be retained. The existing `/fva/waterfall` remains the separately labeled backtest-based naive-baseline view.
- `champion` surfaces first, followed by contender ranks 1, 2, and 3. No fifth AI-adjusted row can enter this archive.

### 6.3 Frontend

New sub-panel `frontend/src/tabs/fva/SnapshotAccuracyPanel.tsx`, mounted in `FVATab.tsx` below the existing waterfall ladder:

- Record-month selector (from `/fva/snapshot-months`) and a four-row model x lag matrix; closed lags show `accuracy_pct` **with the bias sign** (over/under direction has asymmetric cost - stockout vs write-off) and the versus-champion delta as a +/- pts badge; open lags render as pending.
- Every cell/row exposes coverage: `n_dfus`, and `n_dfus_common` on the delta badge; the versus-champion badge is suppressed (rendered as n/a with a tooltip) when the intersection is empty.
  Measured context: 27% of the 12,306 June-forecast DFUs had zero June actuals, so a low lag-0 accuracy without coverage and bias context would mislead.
- A caption shows the MV last-refresh time from `/fva/snapshot-months`.
- `champion` pinned first; contenders are labeled and ordered `#1` through `#3` from the frozen roster.
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
  lag_count: 6             # fixed archive scope: lags 0..5; startup validates this value
  contender_count: 3       # fixed archive scope: champion plus exactly three contenders
  rank_metric: wape        # ascending aggregate WAPE from completed, loaded backtest_run
  active_window_months: 12 # DFU must have an actual within this window to be scored
```

## 8. Testing

| Test | Covers |
|---|---|
| `tests/unit/test_forecast_snapshot_schema.py` | sql/202 structural contract: roster role/rank constraints, source-run and backtest-run foreign keys, and generated `lag` constrained to 0..5 |
| `tests/unit/test_prepare_forecast_snapshot_contenders.py` | Deterministic top-three WAPE selection, tie-breaks, no eligible fourth model, frozen roster immutability, generated run ids, and six-month contender generation |
| `tests/unit/test_archive_forecast_snapshot.py` | Record-month derivation including empty-staging no-op, roster-only source predicates, exact lag-0..5 window, DO NOTHING vs --overwrite, incomplete roster/champion/missing-lag hard failures (mocked psycopg) |
| `tests/unit/test_cleanup_forecast_staging.py` | Selected-generation reconciliation, required champion row, deliberate exclusion of unselected models and lags >5, default targeting, and dry-run |
| `tests/api/test_fva.py` (extend) | `/fva/snapshot-accuracy` four-row matrix, roster metadata, common-DFU champion delta, null-KPI, required `record_month`, lag validation, and empty archive; `/fva/snapshot-months` via `make_pool` |
| `tests/api/test_backtest_management.py` (extend) | promote_model plan_version now derived from `get_planning_date()` |
| `tests/unit/test_mv_refresh.py` (existing, automatic) | Fails unless `agg_accuracy_snapshot` is registered in `MV_SOURCES` with sources matching the DDL |
| `frontend/src/tabs/__tests__/FVATab.test.tsx` (extend) + `fva/SnapshotAccuracyPanel.test.tsx` | Four-row matrix rendering, contender-rank labels, pending lags, bias sign, coverage badges, and suppressed delta on empty intersection |
| `tests/unit/test_pipeline_presets.py` (existing, automatic) | forecast-publish's leading archive and trailing contender-preparation steps reference registered job types in the required order |

## 9. Migration & rollout

1. `sql/202_create_forecast_snapshot.sql`: roster table, archive table, MV, constraints, and indexes (idempotent `IF NOT EXISTS` per repo convention); `MV_SOURCES` Tier-1 entry in the same commit. (`sql/201` is already the latest migration.)
2. `promote_model()` plan_version source fix (4.2 prerequisite) and the generation `--run-id` support in the same change, with tests.
3. Add `forecast_snapshot_roster` and `fact_forecast_snapshot` to the `db-truncate-data` Makefile transaction and to the verbatim SQL block in `docs/operations-manual/11-maintenance-troubleshooting.md`; document `forecast-snapshot-contenders`, `forecast-archive`, and `forecast-staging-clean` in that doc's Data Cleanup section.
4. Register the contender-preparation and archive job types; add Make targets; prepend archive and append contender preparation to `forecast-publish`; optional disabled `default_schedules` entry.
5. Docs: `docs/ARCHITECTURE.md` fact-table catalog; this spec is indexed in `docs/specs/README.md` (row flagged **Proposed** until shipped); add the Implemented Features row only when shipped.

### 9.1 Initial run (June 2026)

1. Apply sql/202 after the already-applied migrations through sql/201.
2. Bootstrap the June roster from the pre-existing June staging rows without regenerating forecasts: choose and freeze ranks 1-3 using the same WAPE rule and the latest runs available no later than the original staging generation. This preserves the true June as-of values and must never train or infer against July data.
3. `make forecast-archive ARGS="--record-month 2026-06"` - expect one champion plus three contender model ids and no rows outside lags 0..5; verify the four per-model counts in the job log against the reconciliation in 4.2.
4. `refresh_for_tables` runs in-script; confirm `agg_accuracy_snapshot` holds lag-0 rows only (June actuals are loaded; July is not).
5. Review lag-0 accuracy in the new FVA panel: it shows exactly four rows, including coverage, bias, and each contender's common-DFU delta versus champion.
6. In the July cycle (planning month 2026-07), run `cleanup_forecast_staging` as its own workflow; the gate verifies only the frozen June roster before deleting June staging.

## 10. Explicitly out of scope

- Archiving `fact_external_forecast_monthly` / `backtest_lag_archive` (owned by 03-backtest-framework.md and cleaned per ../07-user-experience/06-backtest-cleanup.md).
- Archiving a fourth contender, every staged model, `fact_ai_champion_forecast`, or any forecast month outside lags 0..5.
- Re-ranking a record month's contenders after its roster is frozen; correcting a bad roster requires an explicitly audited new record month, not an overwrite of historical selection.
- Dimension-sliced snapshot accuracy (needs a fan-out-safe dim strategy at the item+loc grain; `dim_item` joins are safe when it comes).
- Pre-delete archiving of the outgoing plan inside the promote transaction itself.
  The leading pipeline step in 4.3 provides the archive-before-overwrite guarantee at the orchestration layer; moving it into the promote transaction is a possible hardening later.
- Horizon-conditioned accuracy surfaces (`horizon_months = lag + 1` restriction) - the data is preserved for it (6.4), the UI is not built.

## 11. Adjacent defects noted during design (not addressed here)

- `fact_candidate_forecast` has no writer anywhere in the repo, yet `POST /backtest-management/{model_id}/load` reports it as the target table (the submitted job actually loads `fact_external_forecast_monthly` + `backtest_lag_archive`).
- `generate_production_forecasts.py` `write_forecast()` (direct production-table upsert) is dead code - never called.
- A champion-mode generate stages rows under each true producing model_id but deletes only `WHERE model_id = 'champion'` beforehand, which can strand rows when the champion roster shrinks between runs.
