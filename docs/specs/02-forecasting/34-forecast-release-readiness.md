# 34 - Forecast Release Readiness

| | |
|---|---|
| **Status** | Implemented (post-release planner readiness) |
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
- nine cluster-tuning profiles are stale and the champion has no recorded cluster
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
  generation, promoted cluster assignments, and stale-tuning evidence;
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

Quality reads `fact_external_forecast_monthly` directly. It must not use
`agg_forecast_monthly`, because that view removes `customer_group` and `lag`.
The operational grain is:

```text
item_id + customer_group + loc + startdate + configured execution_lag
```

Rows are joined to `dim_sku` on its full three-column key and retained only when:

1. `model_id` is `champion`, `external`, or `seasonal_naive`;
2. `lag = COALESCE(dim_sku.execution_lag, 0)`;
3. all three model rows exist for the same key; and
4. actual demand is identical across the three rows.

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
| Lift vs seasonal naive | At least 10% relative WAPE improvement | Champion must outperform the simplest baseline |
| Delta vs external | At least 0 accuracy points | Do not replace a better incumbent forecast |
| Champion bias | Absolute bias at most 5% | Avoid service/inventory distortion hidden by aggregate accuracy |
| Actual alignment | Zero mismatched actuals | All model comparisons must use the same truth |
| Active promotion state | Exactly one active promotion with a non-null plan version | Fail closed on missing, ambiguous, or unaudited production ownership |
| Champion-results lineage | Active promotion's experiment is the sole results-promoted experiment, with no champion row modified after `results_promoted_at` | Detect canonical champion rewrites that invalidate the explicit results promotion |
| Cluster lineage | Exactly one promoted cluster experiment matching the champion experiment | Prevent mixed assignments or routing trained under superseded clusters |
| Cluster assignments | At least one current promoted assignment | Prevent per-cluster generation from silently collapsing |
| Generation freshness | Oldest active release row generated at or after latest completed sales load | Prevent an older generated release from looking fresh because it was promoted later |
| Tuning freshness | Zero stale profiles | Current clusters require current per-cluster tuning |
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

## Important boundary

This feature verifies the **already active** release for planner use. It does not
yet transactionally guard `POST /backtest-management/champion/promote` and does
not claim to do so. The current staging table cannot safely express that gate:
its uniqueness key lets contender generation overwrite champion source rows, and
promotion does not select one coherent source run.

Historical `model_id='champion'` rows do not yet carry an experiment ID. The
scorecard therefore requires the active promotion's
`champion_experiment_id` to be the sole `is_results_promoted` experiment, but
also requires the newest champion `modified_ts` to be no later than that
experiment's `results_promoted_at`. This freshness fence detects the canonical
champion-selection rewrite even though it is not a persisted experiment ID or
row checksum. After `model-refresh` ends in champion selection, the operator
must explicitly promote those experiment results before release readiness can
recover. Exact candidate quality evidence and forecast-value checksums belong
in the next transactional phase. Because `model_promotion_log` does not yet
persist the promoted production `run_id`, the post-release archive check is
deliberately described as bounded structural evidence, not an exact value-level
checksum.

The next implementation phase must:

1. make staging run- and purpose-scoped so release candidates and snapshot
   contenders coexist;
2. evaluate a specified candidate `source_run_id` on the primary connection;
3. re-run structural checks inside the promotion transaction before any delete;
4. fail on route gaps instead of warning; and
5. persist the candidate gate report and artifact checksum in the promotion
   audit trail.

Until that phase ships, the card is the fail-closed post-release decision-support
scorecard for whether planning may proceed, but it is not a database constraint
on promotion and does not prove model-training data cutoff or value-level archive
identity.

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
