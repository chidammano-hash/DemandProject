# Full Reset & Reload Runbook — June 2026

**Goal:** Wipe the Postgres database clean **and** delete all generated `data/` artifacts
(preserving `data/input/`), then run the full pipeline to a live, working product for
planning month **2026-06**.

**Scope:** every model (tree + foundation Chronos/Bolt/Chronos2/2e + DL N-BEATS/N-HiTS +
MSTL + baselines), all optional domains (customer demand, sourcing/POs, external ML
extracts), full product depth (forecast → promote → inventory → demand planning → ops).

> Validated against `docs/operations-manual/11-maintenance-troubleshooting.md`,
> the `Makefile`, `config/etl_config.yaml`, `common/core/domain_specs.py`, and
> `config/forecasting/forecast_pipeline_config.yaml`.

---

## One-time PATH fix (avoids the "uv not on PATH under make" trap)

```bash
export PATH="$HOME/.local/bin:$PATH"   # so `make` finds uv
cd /Users/manoharchidambaram/projects/DemandProject
```

If any `make` target still errors on uv, run its underlying `uv run python scripts/...`
command directly.

---

## Required input files (in `data/input/`)

**Required (7) — minimum viable load:**

| File | Format | Feeds |
|---|---|---|
| `itemdata.csv` | CSV | `dim_item` |
| `locationdata.csv` | CSV | `dim_location` (site_id resolution) |
| `customerdata.csv` | CSV | `dim_customer` |
| `dfu.txt` | pipe-delimited | `dim_sku` (sales/forecast filtered against this) |
| `dfu_lvl2_hist.txt` | pipe-delimited | `fact_sales_monthly` (TYPE=1 only) |
| `dfu_stat_fcst.txt` | pipe-delimited | `fact_external_forecast_monthly` (`model_id='external'`) |
| `Inventory_Snapshot_YYYY_MM.csv` | CSV (≥1 month) | `fact_inventory_snapshot` |

(The `time` dimension is auto-generated 2020–2035; no file needed.)

**Optional (in scope for this reload):**

- `*_customer_demand.csv` → customer-analytics MVs (`mv_ca_*`)
- `sourcing.csv` → `dim_sourcing`
- `purchase_orders.csv` → `fact_purchase_orders`
- `df_ml_{lgbm,cat,xg,best}_l2_extract.csv` → external ML competition models (`make load-ext-all`)
- `aws_ml_fcst_*.csv`, `combined_model_map.csv` → **unused by any loader — can be deleted**

---

## PHASE 0 — Prerequisites

```bash
make up                                        # Postgres 16 + Redis + MLflow; also runs db-apply-sql
uv sync --extra foundation --extra dl          # REQUIRED for Chronos (foundation) + N-BEATS/N-HiTS (dl)
# optional: add --extra gpu --extra expert-panel
make db-apply-sql                              # idempotent; skip if schema unchanged.
                                               # Does NOT run sql/184,185 (weekly-partition cutover) — leave excluded.
```

## PHASE 1 — Wipe database + generated files (preserves `data/input/`)

```bash
make db-truncate-data       # ~90 TRUNCATE CASCADE: facts + dims + all history/experiments; keeps config masters
make clean-artifacts        # deletes data/{staged,backtest,tuning,clustering,champion,models,perf_reports}
                            # LEAVES data/input/ untouched
```

## PHASE 2 — Confirm inputs present

```bash
ls -1 data/input/
```

## PHASE 3 — Normalize + load + first MV pass

```bash
make fresh-load             # = normalize-all → load-all → refresh-mvs-tiered  (~5 min)
                            # customer_demand, sourcing, purchase_order auto-handled (files present)
```

## PHASE 4 — SKU features, clustering, lead-time, ABC-XYZ, demand signals

```bash
make features-compute       # SKU features → dim_sku
make cluster-all            # REQUIRED: re-populates dim_sku.ml_cluster (blanked by reload)
make lt-profile-all         # lead-time profiles → feeds safety stock
make abc-xyz-all            # ABC-XYZ classification
make demand-signals-all     # demand signals
```

## PHASE 5 — Backtests, FULL roster  ⏳ LONG POLE — run overnight / in background

```bash
make backtest-all           # lgbm, catboost, xgboost, chronos, bolt, chronos2, chronos2e
make backtest-baselines     # seasonal_naive, rolling_mean (champion fallback)
make backtest-mstl          # statistical
make backtest-nbeats        # DL
make backtest-nhits         # DL
# chronos2/chronos2e ~5.5–6h EACH; nbeats/nhits add hours. Total >> 12h.
# Alt for the tree+foundation set if RAM/GPU allows: make backtest-all-parallel (logs in data/backtest/logs/)
```

## PHASE 6 — Load all backtests + external ML models + accuracy MVs

```bash
make backtest-load-all-bulk # load every model under data/backtest/*/ → fact_candidate_forecast (~4x faster)
make load-ext-all           # ext_lgbm/cat/xg/best → fact_external_forecast_monthly + backtest_lag_archive
make refresh-accuracy-mvs   # 4 accuracy MVs — MUST run AFTER the two loads above
```

## PHASE 7 — Champion selection

```bash
make champion-all           # train-meta → simulate → select; writes data/champion/dfu_assignments.csv + champion rows
```

## PHASE 8 — Production forecast + promote (plan_version = 2026-06)

```bash
make forecast-full          # train-production-all + forecast-generate → fact_production_forecast_staging
make api                    # SEPARATE terminal — promote is an API call, no Make target
# Do NOT set PLANNING_DATE — leave it at today so staging & promote plan_version both = 2026-06.
curl -X POST -H "X-API-Key: $API_KEY" \
  "http://localhost:8000/backtest-management/champion/promote?promoted_by=reset-2026-06"
# model_id=champion BYPASSES the WAPE/coverage gate (experiment-level gating) — cold first promote will NOT be rejected.
# X-API-Key = the API_KEY value in your .env.  → 201 with {"promoted": {...}}
```

## PHASE 9 — Inventory planning, demand planning, operations

```bash
make seed-baselines         # seed production baseline experiment rows (precedes inventory)
make setup-inv-planning     # eoq, policy, ss, exceptions, fill-rate, health, supplier-perf, investment, intramonth, control-tower, rebalancing
make setup-demand-planning  # projection, planned-orders, replplan, quantile, consensus, bias, blended, service-level, lead-time, echelon (consumes PROMOTED forecast)
make setup-ops              # S&OP, events, financial plan, storyboard, scenarios, DQ
```

## PHASE 10 — Final MV refresh + verify

```bash
make refresh-mvs-tiered     # final tiered pass so all downstream/customer MVs reflect promoted forecasts + inventory
make health                 # check-db (row counts) + check-api
make ui                     # Vite on :5173 — open http://localhost:5173
```

---

## Critical notes

- **`make fresh-all` is NOT enough for "fully working."** It stops at champion + *baseline*
  inventory; it does NOT generate/promote production forecasts or run demand/ops planning.
  Phases 8–9 are the gap.
- **Promotion has no Make target** (Phase 8 curl). For `model_id=champion` the WAPE/coverage
  gate is bypassed (`backtest_management.py:923-931`) — no `bypass_token` needed.
- **`backtest-all` ≠ all models** — it's tree + foundation only. The DL/statistical/baseline
  targets in Phase 5 are required for the full roster.
- **Do NOT run `make setup-all`** for this — it re-runs `backtest-all` + `champion-all`,
  redoing the overnight Phase 5. The granular Phase 9 targets avoid that.
- **Ignore manual §9.2 "Option B"** legacy script references (`detect_seasonality.py`, etc.) —
  those scripts were removed; `features-compute` + `cluster-all` replace them.
- **Biggest time risk:** Phase 5 with foundation+DL is >> 12h. Run it detached/overnight.
- **`clean-artifacts` defect (fixed 2026-06-17):** the recipe used a hardcoded list of
  backtest model dirs (`lgbm_cluster`, `chronos`, …) that had drifted to zero overlap with
  the current roster (`bolt_hierarchical`, `lgbm_global`, `mstl`, `nbeats`, `nhits`,
  `seasonal_naive`), so it cleaned nothing under `data/backtest/`. Left stale, Phase 6's
  `backtest-load-all-bulk` (loads every `data/backtest/*/`) would re-ingest last run's
  predictions and corrupt champion selection. Now globs `rm -rf data/backtest/*`.
