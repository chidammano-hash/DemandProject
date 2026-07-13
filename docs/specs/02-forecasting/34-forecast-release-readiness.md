# 34 - Forecast Release Readiness

| | |
|---|---|
| **Status** | Implemented (post-release readiness plus transactional pre-release control) |
| **Date** | 2026-07-10 |
| **API** | `GET /forecast-release/readiness` |
| **UI** | Command Center `ForecastReleaseGateCard` |
| **Related** | [07 Champion Selection](07-champion-selection.md), [24 Candidate Promotion](24-candidate-forecast-promotion.md), [33 Snapshot Archive](33-forecast-snapshot-archive-fva.md) |

## Problem

The platform previously exposed model accuracy, pipeline staleness, production
coverage, and snapshot FVA as separate surfaces. A planner could see a plausible
accuracy number while the champion was built before the latest sales load, had
unknown cluster lineage, lacked a current-month production plan, or had never
been archived. The FVA waterfall also compared stage accuracies whose row counts
and populations differed, so subtracting those values did not prove value added.

The July 2026 `1401-BULK` pilot demonstrates the risk:

- the published production plan is version `2026-06` while the planning month is
  `2026-07`;
- the latest sales load occurred after the promoted champion;
- current cluster-tuning profiles are stale and the champion has no recorded cluster
  experiment lineage;
- `fact_forecast_snapshot` is empty, so the outgoing plan is not preserved; and
- on the configured six-month common cohort, champion WAPE is 29.03% versus
  26.05% for the external forecast.

## Solution

Expose one post-release readiness contract for planner use. It evaluates the
current planning month as a versioned release and returns:

- forecast skill and bias on one exact common observation cohort;
- common-cohort coverage relative to the valid champion population;
- active-promotion champion-results lineage, exactly one promoted cluster
  generation, promoted cluster assignments, and stale-tuning evidence scoped to
  cluster labels used by the current assignment generation;
- latest-sales versus release-generation freshness;
- complete current-plan coverage plus one-run, quantity, source-model, and
  confidence-interval integrity; and
- bounded structural evidence that the outgoing champion-plus-three release was
  archived for lags 0 through 5 during its active lifetime and before replacement.

The Command Center renders the result as a persistent readiness card. It shows
the release version, four headline metrics, evidence population, explicit
blockers, and the next safe workflow to inspect. Empty or stale data is never
rendered as a healthy zero. Six blockers render initially; an accessible
disclosure exposes the complete list. The card re-evaluates every 60 seconds
while it remains open, including after a green result.

## Exact common cohort

Quality reads the active champion and, when required by policy, external rows
from `fact_external_forecast_monthly` directly and derives the seasonal-naive
baseline from `fact_sales_monthly` at the matching DFU's month twelve months
earlier. It does not require a retained `seasonal_naive` algorithm series and
must not use `agg_forecast_monthly`, because that view removes
`customer_group` and `lag`. The operational grain is:

```text
item_id + customer_group + loc + startdate + configured execution_lag
```

Rows are joined to `dim_sku` on its full three-column key and retained only when:

1. a `champion` row's `champion_experiment_id` equals the active promotion's
   explicit champion experiment;
2. `lag = COALESCE(dim_sku.execution_lag, 0)`;
3. the external row exists for the same key when
   `champion.release_readiness.require_external_benchmark` is `true`;
4. the seasonal-naive value is taken from type-1 sales for that DFU exactly
   twelve months earlier, with an absent sparse-fact month densified to zero;
   and
5. champion and external actual demand are identical when the external benchmark
   is required. The derived baseline always inherits the champion actual.

When `require_external_benchmark` is `false`, the exact cohort contains champion
and seasonal-naive rows only. The external accuracy-delta check reports `pass`
with threshold `not required`; all six-month, coverage, DFU-count, actual-volume,
naive-lift, bias, lineage, confidence-interval, and archive checks remain enforced.
The `1401-BULK` laptop pilot uses this explicit exemption until its external feed
contains the current closed month and representative DFU coverage. Re-enable the
flag after that feed is loaded; do not synthesize or forward-fill external rows.

A genuine prior-year zero and an omitted zero-demand month both produce a zero
baseline, matching the dense monthly history used by forecasting. The sales
scan is bounded to the exact prior-year months required by the active champion
cohort.

Pre-migration champion rows with NULL experiment lineage are therefore excluded
rather than silently attributed to the active release.

Portfolio metrics are calculated from summed components:

```text
WAPE%    = 100 * SUM(ABS(forecast - actual)) / ABS(SUM(actual))
Accuracy = 100 - WAPE%
Bias%    = 100 * (SUM(forecast) / SUM(actual) - 1)
```

`source_model_id` is not a grouping dimension. For `model_id='champion'` it
records the routed underlying winner, while the portfolio series remains one
champion.

The lookback is fixed by policy and cannot be weakened with a request query
parameter. In addition to non-empty overlap, the cohort must span all six
closed months, contain at least 1,000 DFUs, and have positive aggregate actual
demand. These are provisional pilot sufficiency floors, not statistical claims
that every segment is individually representative.

## Readiness checks

| Check | Current policy | Why it matters |
|---|---|---|
| Readiness policy | Enabled; disabled fails closed | Turning off evaluation must never imply safety |
| Common cohort | At least one valid observation | No quality claim without measurable overlap |
| Cohort coverage | At least 95% of valid champion observations | Prevent a small favorable intersection from looking representative |
| Closed months | All six configured months | Prevent a favorable one-month slice from approving a release |
| Cohort DFUs | At least 1,000 | Prevent a tiny favorable population from approving a portfolio release |
| Actual volume | Greater than zero | Avoid unstable/null WAPE and bias evidence |
| Delta vs external | At least 0 accuracy points when `require_external_benchmark: true`; otherwise explicitly `not required` | Do not replace a better incumbent forecast when comparable external evidence is available |
| Champion bias | Absolute bias at most 5% | Avoid service/inventory distortion hidden by aggregate accuracy |
| Actual alignment | Zero mismatched actuals | All model comparisons must use the same truth |
| Active promotion state | Exactly one active promotion with a non-null plan version | Fail closed on missing, ambiguous, or unaudited production ownership |
| Champion-results lineage | Active promotion's experiment is the sole results-promoted experiment, with no champion row modified after `results_promoted_at` | Detect canonical champion rewrites that invalidate the explicit results promotion |
| Cluster lineage | Exactly one promoted cluster experiment matching the champion experiment | Prevent mixed assignments or routing trained under superseded clusters |
| Cluster assignments | At least one current promoted assignment | Prevent per-cluster generation from silently collapsing |
| Sales-source lineage | Populated immutable-original mirror; latest positive completed audit is a canonical dual-track reload (not `safe_upsert`); mirror row count matches the batch and mirror `MAX(load_ts) >= batch.started_at` | Prevent current/original divergence or an audit timestamp written after its rows from being treated as synchronized evidence |
| Generation freshness | Oldest active release row generated at or after the latest accepted synchronized sales load | Prevent an older generated release from looking fresh because it was promoted later |
| Tuning freshness | Zero stale profiles for labels in the current promoted assignment generation | Current clusters require current per-cluster tuning without historical labels blocking release |
| Current plan version | Equals the planning month | Make release cadence explicit |
| Current plan coverage | At least 95% of active DFUs with 3+ history months and all six calendar months | Ensure the plan is operationally usable |
| Release integrity | One run ID; no negative/invalid quantities; complete source-model lineage; valid confidence intervals on at least 95% of rows | Prevent count-complete but internally inconsistent inventory inputs |
| Outgoing archive | Previous plan has four models by six lags, one champion run, and roster/run/plan lineage archived after its promotion and before replacement | Preserve the plan before replacement without accepting an older same-version snapshot |

Thresholds live under `champion.release_readiness` in
`config/forecasting/forecast_pipeline_config.yaml`. They are provisional pilot
targets, not external benchmarks. They must be revisited after a measured pilot,
without changing the canonical formulas or grain.

## Coverage denominator

An eligible item-location has at least
`production_forecast.cold_start_min_months` distinct sales months and a most
recent sale inside `forecast_snapshot.active_window_months`. A complete plan has
exactly `forecast_snapshot.lag_count` distinct calendar months inside the fixed
planning-month window. This excludes discontinued DFUs and forecasts that only
appear complete because they include months outside the release horizon.
The same fixed-window rows must belong to one release `run_id`, have nonnegative
point/bound quantities, carry `source_model_id`, maintain lower <= point <=
upper when intervals are present, and meet the configured interval-coverage
floor.

## API contract

`GET /forecast-release/readiness` returns:

```json
{
  "ready": false,
  "policy_enabled": true,
  "release_version": "2026-07",
  "quality": {
    "dfu_months": 41646,
    "closed_months": 5,
    "common_observation_coverage_frac": 0.9396,
    "champion_accuracy_pct": 70.966,
    "champion_bias_pct": -0.501,
    "relative_wape_lift_vs_naive_pct": 12.218,
    "accuracy_delta_vs_external_pct_points": -2.983
  },
  "checks": [
    {
      "id": "delta_vs_external",
      "status": "block",
      "value": -2.983,
      "threshold": 0.0,
      "message": "Champion accuracy trails the external forecast on the common cohort."
    }
  ],
  "next_action": {
    "tab": "dataQuality",
    "pipeline": null,
    "label": "Review forecast data quality"
  }
}
```

The endpoint is a Pydantic-v2-typed, cached, read-only planner scorecard. Its
four evidence queries execute in one `REPEATABLE READ, READ ONLY` transaction so
concurrent loads or promotions cannot create a mixed-time verdict. Cache TTL is
60 seconds; the Command Center polls every 60 seconds while open so a green card
cannot survive a newer load, promotion, or planning-month rollover.
Database errors produce an opaque 500.
Freshness uses the minimum `generated_at` across the active champion release,
not the champion-experiment or administrative promotion timestamp; a later
promotion cannot make rows generated from older inputs fresh.
The UI action navigates to the surface that can safely resolve the blocker. It
does not claim to preselect or execute the named pipeline: refresh actions open
Jobs, while archive recovery opens the FVA evidence surface.
Archive readiness follows Spec 33's outgoing-before-replacement invariant. It
checks the promotion immediately preceding the active promotion and only
accepts snapshot rows archived from that outgoing promotion timestamp through
the replacement timestamp. Champion `plan_version`/run and contender
frozen-roster run IDs must match. A first-ever release has no outgoing plan to
archive; a newly active release is not blocked merely because its own future-FVA
snapshot is not due until the next replacement cycle.

## Transactional promotion boundary

The follow-on transactional phase is implemented by migration
`sql/203_create_forecast_generation_run.sql` and
`common/services/forecast_promotion.py`. The Command Center card remains the
read-only, **post-release** planner scorecard described above; promotion now has
a separate fail-closed, **pre-release** contract:

The Forecast stage exposes that pre-release contract through two focused reads:
`GET /backtest-management/training-status` validates persisted model artifacts
against current sales, history, generator, cohort/runtime, configuration, and
cluster lineage, while `GET /backtest-management/snapshot-roster-readiness`
deeply validates the current champion plus exact top-three contender evidence.
The UI never treats artifact self-metadata or a staging count as sufficient.
Promotion remains disabled until both reads and the current champion candidate
are ready. Its **Prepare Release** action launches the named `forecast-publish`
pipeline and polls its exact pipeline id; integrity-corrupt evidence fails
closed for operator investigation instead of advertising a rebuild that cannot
safely replace it.

1. Every generation has one immutable `forecast_generation_run` manifest and a
   purpose: `release_candidate`, `snapshot_contender`, or `legacy_invalid`.
   Normal champion generation produces one coherent `release_candidate` run.
   Its staged rows keep the requested candidate id (`champion`) separately from
   each routed `model_id`, so the exact source model is preserved without mixing
   another generation into the candidate.
2. `POST /backtest-management/{model_id}/promote` requires `source_run_id` and
   promotes only that run. Pre-migration staging is classified
   `legacy_invalid` and is deliberately not promotable; a fresh generation is
   required after applying migration 203. The chosen champion experiment's
   results must also be explicitly promoted again so historical rows receive
   the experiment id and exact artifact/result checksums before generation.
3. Promotion uses the primary connection in one `SERIALIZABLE` transaction and
   obtains the transaction-scoped advisory lock
   `forecast_release_promotion`. It locks the source manifest, re-computes the
   canonical staging checksum, verifies the exact experiment-stamped historical
   champion-result payload, re-evaluates the common-cohort quality policy, and
   re-runs the structural, lineage, freshness, coverage, route-gap, quantity,
   and confidence-interval checks. It performs no production delete until those
   checks pass.
4. If an active outgoing release exists, the transaction archives its exact
   champion plus the frozen top-three `snapshot_contender` runs for lags 0
   through 5 before demotion or delete. It reconciles the champion archive to
   the outgoing production payload checksum. Any incomplete roster, lag, run
   lineage, or checksum aborts the entire promotion.
5. The new production payload is hashed in stable business-key order and must
   equal the selected candidate checksum. `model_promotion_log` persists
   `source_run_id`, a distinct `production_run_id`, the gate report, candidate
   and production checksums, replacement lineage, and the outgoing champion
   archive checksum. Database indexes enforce one active promotion and prevent
   the same source run from being promoted twice.

The transaction re-evaluates WAPE/lift/incumbent/bias thresholds against the
exact historical champion rows stamped with the promoted experiment id and
stores the resulting checks in `gate_report`. It intentionally does **not**
stamp a "candidate WAPE" on the forward generation: future candidate rows have
no actuals, so such a number would be false precision. Migration 203 stores the
promoted winners-artifact checksum plus the checksum/row count of the exact
experiment-stamped historical champion payload; generation captures those
hashes and promotion re-computes them before evaluating quality.

Pre-migration champion rows still lack a row-level experiment id and are not
silently attributed to an experiment. A new results-promotion job loads the
cached winners with `champion_experiment_id`, hashes the resulting historical
rows, and only then marks the experiment results promoted. The post-release
scorecard also retains its `modified_ts <= results_promoted_at` freshness fence.
The final `model-refresh` step is a governed champion refresh. Before creating
an experiment it requires the latest completed-and-loaded backtest for each of
the five models to share the current sales batch/hash and promoted cluster
experiment/assignment checksum. It evaluates the current production strategy
while leaving the incumbent active, then loads the exact winners artifact,
records checksums, and swaps champion facts plus configuration/results
promotion flags in one transaction. A failure cannot erase the incumbent.
The operator can proceed directly to generating a new release candidate; a
separate experiment-results promotion is not required for the named workflow.

## Tests

- `tests/unit/test_forecast_release.py` pins formulas, common-cohort coverage
  and sufficiency, baseline lift, incumbent regression, bias, and empty-cohort
  behavior.
- `tests/api/test_forecast_release.py` pins the API contract for all-pass and
  multi-blocker releases, fixed policy window, fail-closed switches, active
  promotion/results/cluster cardinality lineage, repeatable-read isolation,
  release integrity, archive lifetime/lineage, custom versions, actionable
  destinations, and typed OpenAPI schema using the project pool factory.
- `frontend/src/components/__tests__/ForecastReleaseGateCard.test.tsx` pins the
  blocked and planner-ready UI states, CTA destination, and full blocker
  disclosure.
- `tests/unit/test_forecast_promotion.py`,
  `tests/api/test_forecast_promotion.py`, and
  `tests/unit/test_forecast_promotion_schema.py` pin explicit-run selection,
  manifest/checksum reconciliation, fail-closed gate errors, serializable
  locking, atomic outgoing archive, rollback, and migration constraints.
