## sql/ + config/ — Refactor Opportunities

_Scope: `sql/` (146 numeric-prefixed DDL migrations) and `config/` (43 YAML files). Read-only audit — no code changed._

### Quick wins
- `config/forecasting/elasticity_config.yaml` is orphaned — its only consumer `scripts/ml/fit_elasticity.py` does not exist; no `load_config`, no whitelist entry. Delete it.
- `sql/88_backtest_run.sql` has an unpadded prefix; under `db-apply-sql`'s `ls sql/*.sql | sort` it lands LAST (after `187_*`). Rename to a zero-padded 3-digit prefix.
- Six duplicate numeric prefixes: `039`, `041`, `090`, `091`, `092`, `095` each used by two unrelated migrations. Renumber the later one in each pair.
- `config/forecasting/tuning_templates.yaml:8` header cites `api/routers/forecasting/unified_model_tuning.py` — that module no longer exists (now the `tuning/` package). Fix the comment.
- `config/forecasting/tune_strategies.yaml:5,16` reference the DELETED `algorithm_config.yaml` as a live promotion target. Update to `forecast_pipeline_config.yaml`.
- `config/ai/agent_autonomy.yaml` has inline comments on only 4 of 47 key lines — violates the "every key documented" rule.
- `dim_sku` columns are added across 6 migrations (`005,015,022,031,120,145`) — squash candidate for a future schema-reset.
- `sql/039_create_production_forecast.sql` sorts AFTER `sql/039_create_ai_call_log.sql` (alpha tiebreak); verify no FK/dependency ordering hazard before renumbering.

### Ranked opportunities

1. **Migration filename prefix collisions + one unpadded prefix break apply-order determinism**
   - Files: `sql/88_backtest_run.sql`; `sql/039_*` (×2), `041_*` (×2), `090_*` (×2), `091_*` (×2), `092_*` (×2), `095_*` (×2); apply target `Makefile:344` (`for f in $(ls sql/*.sql | sort)`)
   - Problem: `db-apply-sql` orders purely by lexicographic filename sort. `88_*` sorts after `187_*` so it applies dead-last on a fresh DB. Six duplicate-prefix pairs apply in alpha-suffix (not authoring) order — any cross-pair dependency is order-fragile.
   - Proposed change: Rename `88_backtest_run.sql` to a free 3-digit prefix; renumber the second file in each duplicate pair. One migration per prefix. Audit each pair for inter-file dependencies first.
   - Impact: High (fresh-DB rebuilds are non-deterministic today) · Effort: Low · Risk: L-M (re-check FK/MV references; files are idempotent `IF NOT EXISTS`)

2. **`fact_safety_stock_targets` accreted across 15 migrations — prime squash candidate**
   - Files: `sql/037_create_safety_stock_targets.sql` (base) + `030,041,073,117,126,129,130,131,133,134,135,136` (+ reads in `026,032`)
   - Problem: The most-altered table — 12 follow-on migrations add columns/indexes/drops; true shape requires reading 13 files.
   - Proposed change: At the next squash/baseline migration, fold into one canonical `CREATE TABLE` + indexes. Short term, add a header comment in `037` listing the follow-ons.
   - Impact: High (readability, onboarding) · Effort: M · Risk: Low (do as an additive baseline; don't reorder applied migrations on live DBs)

3. **Orphaned config file: `elasticity_config.yaml`**
   - Files: `config/forecasting/elasticity_config.yaml`; missing `scripts/ml/fit_elasticity.py`; stale allowlist ref `scripts/ai_checks/allowlists/rule6_fstring_sql.txt:50`
   - Proposed change: Delete the YAML + the stale allowlist line.
   - Impact: Med · Effort: Low · Risk: Low (confirmed no loader)

4. **Stale references to deleted `algorithm_config.yaml` in tuning configs**
   - Files: `config/forecasting/tune_strategies.yaml:5,16`; `tuning_templates.yaml:15,32,100,170`
   - Problem: `algorithm_config.yaml` was deleted (folded into `forecast_pipeline_config.yaml`), but comments still describe promoting params "into algorithm_config.yaml". Note: `source: algorithm_config` in `tuning_templates.yaml` is a LIVE marker (`tuning/templates.py:45`), so don't blanket-rename — only the comments are misleading.
   - Proposed change: Update comments to `forecast_pipeline_config.yaml`. Optionally rename the `source:` marker to `pipeline_config` and update `tuning/templates.py` in the same change.
   - Impact: Med · Effort: Low (comments) / M (marker) · Risk: Low / Med

5. **Three overlapping tuning configs with unclear boundaries**
   - Files: `hyperparameter_tuning.yaml`, `tune_strategies.yaml`, `tuning_templates.yaml` (+ `cluster_tuning_profiles.yaml`)
   - Problem: Four tuning YAMLs with distinct loaders but overlapping scope (search spaces, per-model overrides, UI templates); no cross-reference, easy to mis-place a key.
   - Proposed change: Add a "Relationship to other tuning configs" header in each. Don't merge (loaders differ) — clarify.
   - Impact: Med · Effort: Low · Risk: Low

6. **`shared_constants.yaml` "distribution paths" duplicate every canonical value into many nested paths**
   - Files: `config/shared_constants.yaml` (anchors `*sla`, `*ztable`, `*fin`, `*ss_rails` re-emitted under 6 consumer paths)
   - Problem: Each consumer expects values at its own key path, so anchors are repeated ~6× via deep-merge; a typo in a path silently yields defaults.
   - Proposed change: Longer-term, normalize consumers to read one shared key path so the distribution duplication can be deleted. Short term acceptable — flag for the next consumer-config refactor.
   - Impact: Med · Effort: M-H · Risk: M (silent fallback-to-default; needs resolved-value tests)

7. **Two backtest tables with overlapping intent: `backtest_run` vs `ai_fva_backtest`**
   - Files: `sql/88_backtest_run.sql`, `186_create_ai_fva_backtest.sql`, `100_results_promotion.sql`, `121_candidate_forecast_and_promotion.sql`
   - Proposed change: Document each table's distinct purpose in a header; if `ai_fva_backtest` duplicates `backtest_run` columns, consolidate or FK-link rather than parallel-store.
   - Impact: Med · Effort: M · Risk: M (live result tables — consolidate only with migration + backfill)

8. **Inconsistent migration application style in the Makefile**
   - Files: `Makefile:344` (generic loop) vs `:515-651,1019` (dozens of one-off `$(PSQL) < sql/NNN_*.sql` and inline `python -c "...open('sql/NNN...')"` with hardcoded host/port/creds)
   - Proposed change: Standardize on `db-apply-sql` (or a small helper); delete per-file targets that merely re-apply one migration; centralize credentials.
   - Impact: Med (removes drift + embedded creds) · Effort: M · Risk: L-M (verify no setup flow depends on a target)

9. **Sparse inline comments in several configs violate the documentation rule**
   - Files: `config/ai/agent_autonomy.yaml` (4/47 commented); spot-check `config/ai/`, `config/operations/`
   - Proposed change: Backfill inline comments + `# ═══ SECTION ═══` headers, starting with `agent_autonomy.yaml`.
   - Impact: Low-Med · Effort: L-M · Risk: None

10. **Stub tables that may never have received real data**
    - Files: `sql/026_create_inventory_health_score.sql`, `037_create_safety_stock_targets.sql`, `067_create_external_signals.sql`, `119_concurrent_mv_refresh.sql`, `184_partition_inventory_snapshot_weekly_cutover.sql`
    - Problem: The stub-table pattern is intentional, but stubs that never got a real producer become permanent dead weight feeding neutral scores. Verify `external_signals` (067) + the health-score stub (026) have live writers.
    - Proposed change: Audit each stub for a producer; wire it or remove the table + its neutral-join consumers.
    - Impact: Low-Med · Effort: M · Risk: M (removing a stub changes LEFT JOIN results — verify scoring first)

**Note:** the "80 ALTERs on dim_sku" signal was a grep artifact (per-column); the real spread is 6 files (#1 quick win). `fact_safety_stock_targets` (#2) is the genuine high-churn table.
