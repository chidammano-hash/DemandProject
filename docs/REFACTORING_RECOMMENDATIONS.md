# Demand Studio — Codebase Refactoring Recommendations

## Executive Summary

The Demand Studio codebase delivers 18+ features across a full-stack analytics platform, but three files concentrate **~8,300 lines** of logic with significant duplication, no modularization, and escalating coupling. This document provides a prioritized refactoring roadmap to reduce footprint, eliminate duplication, and establish patterns that scale beyond MVP.

**Estimated removable duplication: ~3,800 lines (30% of total codebase)**

---

## Current State Snapshot

| File | Lines | Role | Maintainability |
|------|-------|------|-----------------|
| `api/main.py` | 2,706 | Entire backend — 37 routes, 41 helpers | 3/10 — god file |
| `frontend/src/App.tsx` | 2,814 | Entire frontend — 40 useState, 19 useEffect | 3/10 — monolith |
| `scripts/run_backtest.py` | 846 | LGBM backtest | 4/10 — 95% duplicated |
| `scripts/run_backtest_catboost.py` | 836 | CatBoost backtest | 4/10 — 95% duplicated |
| `scripts/run_backtest_xgboost.py` | 831 | XGBoost backtest | 4/10 — 95% duplicated |
| `scripts/run_backtest_prophet.py` | 751 | Prophet backtest | 4/10 — 90% duplicated |
| `common/domain_specs.py` | 429 | Domain definitions (7 domains) | 9/10 — excellent |
| `Makefile` | 397 | 72 build targets | 7/10 — some duplication |
| **All scripts total** | **6,350** | 14 Python scripts | |
| **Grand total** | **~12,700** | | |

---

## R1. Backtest Scripts — Extract Common Framework

**Priority: HIGH | Effort: MEDIUM | Risk: LOW**
**Impact: ~2,400 lines eliminated (72% reduction in backtest code)**

### Problem

Four backtest scripts (LGBM, CatBoost, XGBoost, Prophet) total **3,264 lines** with 95% identical code. Only the model training functions differ (~30-80 lines per framework). Adding a new model requires copying 800+ lines and risks divergence.

### Duplication Evidence

Exhaustive function-by-function comparison across all 4 scripts:

| Function | Per Script | Scripts | Identical? | Total Waste |
|----------|-----------|---------|------------|-------------|
| `get_db_conn()` | 8 lines | 4 | YES — byte-for-byte | 24 lines |
| `_ts()` | 3 lines | 4 | YES | 9 lines |
| `generate_timeframes()` | 26 lines | 4 | YES | 78 lines |
| `build_feature_matrix()` | 86 lines | 3 (not Prophet) | YES except L176 dtype | 172 lines |
| `get_feature_columns()` | 4 lines | 3 | YES | 8 lines |
| `mask_future_sales()` | 20 lines | 3 | YES | 40 lines |
| `assign_execution_lag()` | 30 lines | 4 | YES | 90 lines |
| `expand_to_all_lags()` | 30 lines | 4 | YES | 90 lines |
| Data loading SQL block | 30 lines | 4 | YES — identical queries | 90 lines |
| Output CSV formatting | 45 lines | 4 | YES — identical columns | 135 lines |
| Accuracy computation | 30 lines | 4 | YES — same WAPE/bias | 90 lines |
| MLflow logging block | 44 lines | 4 | YES — identical flow | 132 lines |
| Argparse setup | 20 lines | 4 | SIMILAR — model params differ | 40 lines |
| Timeframe train loop | 60 lines | 4 | YES — identical orchestration | 180 lines |
| **Total duplicated** | **~436** | | | **~1,178 lines** |

The **only** code that genuinely differs per model:

| Function | LGBM | CatBoost | XGBoost | Prophet |
|----------|------|----------|---------|---------|
| `train_and_predict_global()` | `lgb.LGBMRegressor` | `cb.CatBoostRegressor` | `xgb.XGBRegressor` | Per-DFU Prophet fit |
| `train_and_predict_per_cluster()` | Same + per-cluster split | Same + `cat_indices` | Same (simpler) | Per-cluster Prophet |
| `train_and_predict_transfer()` | `init_model=booster_` | `init_model=base_model` | `xgb_model=get_booster()` | Pooled aggregation |
| Categorical handling | `.astype("category")` | `.astype(str)` + indices | `.astype("category")` | N/A |
| Hyperparams | n_estimators, num_leaves | iterations, depth | max_depth, subsample | seasonality, growth |

**Each model's unique code: ~80-150 lines out of ~830 total.**

### Recommended Structure

```
common/
├── db.py                    (~15 lines — get_db_params, shared across 6+ scripts)
├── backtest_framework.py    (~400 lines — orchestrator with all shared logic)
├── feature_engineering.py   (~120 lines — build_feature_matrix, lag/rolling features)
├── metrics.py               (~30 lines — WAPE, accuracy, bias formulas)
├── mlflow_utils.py          (~50 lines — experiment logging wrapper)
├── constants.py             (~20 lines — LAG_RANGE, ROLLING_WINDOWS, output columns)
scripts/
├── run_backtest.py          (~130 lines — LGBM adapter: train functions + hyperparams)
├── run_backtest_catboost.py (~130 lines — CatBoost adapter only)
├── run_backtest_xgboost.py  (~130 lines — XGBoost adapter only)
├── run_backtest_prophet.py  (~200 lines — Prophet adapter, multiprocessing pool)
```

### Key Actions

1. **Create `common/backtest_framework.py`** — Abstract orchestrator implementing: data load → feature build → timeframe loop → strategy dispatch → output → MLflow. Each model registers a training adapter via a simple interface:
   ```python
   class ModelAdapter:
       def train_global(self, train_df, predict_df, feature_cols, cat_cols, params) -> pd.DataFrame
       def train_per_cluster(self, ...) -> pd.DataFrame
       def train_transfer(self, ...) -> pd.DataFrame
   ```
2. **Create `common/feature_engineering.py`** — Move `build_feature_matrix()`, `mask_future_sales()`, `get_feature_columns()`, constants (`CAT_FEATURES`, `LAG_RANGE`, `ROLLING_WINDOWS`), and `forecast_ck` construction
3. **Create `common/db.py`** — Move `get_db_conn()` (currently duplicated in 6 scripts)
4. **Create `common/metrics.py`** — Move WAPE/accuracy/bias computation (duplicated in 4 scripts + API)
5. **Create `common/mlflow_utils.py`** — Extract identical MLflow logging pattern

### Result

| Metric | Before | After |
|--------|--------|-------|
| Lines across 4 scripts | 3,264 | ~590 |
| Adding a new model | Copy 830 lines | Write ~130 lines |
| Feature engineering changes | Edit 3 files | Edit 1 file |
| Metrics formula changes | Edit 4 scripts + API | Edit 1 file |

---

## R2. Backend API — Break the God File

**Priority: HIGH | Effort: MEDIUM | Risk: MEDIUM**
**Impact: ~900 lines eliminated (35% reduction in API code)**

### Problem

`api/main.py` is **2,706 lines** containing **37 route handlers** and **41 helper functions** in a single file. It mixes HTTP routing, business logic, raw SQL queries, external API integration (OpenAI, Google), file I/O, and YAML config management. Every feature addition touches this one file.

### Specific Issues

**14 copy-pasted backward-compatible endpoints (lines 2203-2319):**
- 7 `GET /{domain}` endpoints (lines 2203-2236) — each is a 5-line wrapper calling `list_domain(get_spec("X"), limit)`
- 7 `GET /{domain}/page` endpoints (lines 2238-2319) — each is an 11-line wrapper calling `fetch_page()` with only the default `sort_by` differing
- **77 lines** of pure duplication that could be a single parameterized handler

**Accuracy filter duplication (~250 lines):**
- `forecast_accuracy_slice()` (lines 855-1115) and `forecast_accuracy_lag_curve()` (lines 1116-1340) share identical filter-building logic for cluster_assignment, supplier_desc, abc_vol, region, month_from, month_to, and common-DFUs CTE intersection

**23 occurrences of `with get_conn()`** — each endpoint manages its own connection lifecycle with inline SQL

**No middleware:**
- No CORS middleware (frontend at :5173 requires it)
- No request logging
- No exception handler middleware
- No connection pool shutdown handler (pool lazily created on line 30 but never closed)

**2 global mutable singletons** (`_pool`, `_openai_client`) with no lifecycle management

### Recommended Structure

```
api/
├── main.py              (~60 lines — app factory, lifespan, middleware, router mounts)
├── deps.py              (~40 lines — pool lifecycle, get_conn dependency)
├── models.py            (~100 lines — Pydantic response schemas)
├── routers/
│   ├── domains.py       (~350 lines — generic /domains/{domain}/* CRUD + page + suggest)
│   ├── accuracy.py      (~300 lines — /forecast/accuracy/slice + lag-curve, shared filters)
│   ├── competition.py   (~200 lines — /competition/config + run + summary)
│   ├── dfu_analysis.py  (~180 lines — /dfu/analysis + clusters + profiles)
│   ├── chat.py          (~200 lines — /chat + OpenAI + pgvector)
│   ├── market_intel.py  (~150 lines — /market-intelligence + Google API)
│   └── bench.py         (~80 lines — /bench/compare latency testing)
├── services/
│   ├── kpi.py           (~40 lines — _compute_kpis, accuracy/WAPE/bias formulas)
│   ├── filters.py       (~100 lines — build_where, _typed_eq_clause, _typed_like_clause)
│   └── query.py         (~40 lines — qident, dotted_qident, identifier helpers)
```

### Key Actions

1. **Extract `deps.py`** — Move `_pool`, `_get_pool()`, `get_conn()` into a dependency module. Add FastAPI `lifespan` context manager for proper pool startup/shutdown
2. **Create FastAPI routers** — `APIRouter(prefix=..., tags=[...])` per feature domain
3. **Eliminate 14 backward-compat endpoints** — Replace with a single `GET /domains/{domain}/page` handler using `get_spec()` dynamically (the generic handler at line 448 already exists — remove the 14 aliases)
4. **Deduplicate accuracy filter logic** — Extract shared filter builder used by both `slice` and `lag-curve` into `services/filters.py`
5. **Extract KPI formulas** — `_compute_kpis()` into `services/kpi.py` (shared with `common/metrics.py` from R1)
6. **Add middleware** — CORS, structured logging, global exception handler

### Result

| Metric | Before | After |
|--------|--------|-------|
| main.py lines | 2,706 | ~60 (app factory only) |
| Total API lines | 2,706 | ~1,840 across 12 files |
| Duplicate endpoints | 14 | 0 |
| Connection lifecycle | No shutdown | Proper lifespan management |
| Testability | Untestable god file | Per-router unit tests |

---

## R3. Frontend — Decompose the Monolith

**Priority: HIGH | Effort: HIGH | Risk: MEDIUM**
**Impact: ~700 lines eliminated (25% reduction), massive testability gain**

### Problem

`App.tsx` is **2,814 lines** — the entire application in a single React component with **40 useState hooks**, **19 useEffect hooks**, and **22 unique fetch endpoints**. Every state change re-renders the entire application. Zero component extraction beyond shadcn/ui primitives.

### Specific Issues

**State explosion — 40 useState hooks in one component:**
- Explorer state: 14 hooks (domain, meta, rows, total, offset, limit, search, sort, filters, columns, etc.)
- Accuracy state: 9 hooks (sliceGroupBy, sliceLag, sliceModels, sliceData, lagCurve, etc.)
- DFU Analysis state: 10 hooks (dfuItem, dfuLocation, dfuPoints, dfuKpiMonths, dfuTimeStart, etc.)
- Chat state: 4 hooks (chatOpen, chatInput, chatMessages, chatLoading)
- Market Intel state: 5 hooks (miItemFilter, miLocationFilter, miLoading, miError, etc.)
- Competition state: 3 hooks (competitionConfig, runningCompetition, savingConfig)

**Repeated fetch pattern — 19 useEffect hooks with identical boilerplate:**
```typescript
useEffect(() => {
  let cancelled = false;
  async function load() {
    try {
      const res = await fetch(...);
      if (!res.ok) throw new Error(...);
      const data = await res.json();
      if (!cancelled) setState(data);
    } catch { if (!cancelled) setError(...); }
  }
  load();
  return () => { cancelled = true; };
}, [deps]);
```
This pattern appears **19 times** with minor variations.

**Copy-pasted JSX — 5 identical tab buttons (lines 1188-1303):**
Each tab renders the same chemistry-element tile structure:
```jsx
<button className={cn(...)}>
  <span className="text-[9px]...">{el.number}</span>
  <span className="text-xl font-black...">{el.symbol}</span>
  <span className="text-[9px]...">{el.name}</span>
</button>
```

**Other duplicated UI patterns:**
- KPI cards: ~15+ instances of the same card layout
- Loading overlays: 3 identical periodic-table loading animations
- Filter input + datalist: 3+ instances
- Model wins bar charts: 2 identical instances
- Checkbox toggle labels: 2+ instances

**Positive: TypeScript is excellent** — 18 well-defined types, zero `any` usage.

### Recommended Structure

```
frontend/src/
├── App.tsx                   (~120 lines — layout shell, tab router, theme provider)
├── api/
│   └── client.ts             (~60 lines — fetch wrapper with error handling + cancellation)
├── hooks/
│   ├── useFetch.ts           (~40 lines — generic data fetching hook)
│   ├── useExplorer.ts        (~80 lines — explorer tab state + effects)
│   ├── useAccuracy.ts        (~60 lines — accuracy tab state + effects)
│   ├── useDfuAnalysis.ts     (~50 lines — DFU analysis state + effects)
│   └── useDebounce.ts        (~15 lines — extract from inline implementation)
├── components/
│   ├── TabButton.tsx          (~35 lines — replaces 5 IIFE copies)
│   ├── KpiCard.tsx            (~25 lines — replaces 15+ inline templates)
│   ├── LoadingOverlay.tsx     (~30 lines — replaces 3 inline copies)
│   ├── ModelWinsBar.tsx       (~30 lines — replaces 2 inline copies)
│   ├── FilterInput.tsx        (~25 lines — input + datalist pattern)
│   └── SeriesToggle.tsx       (~20 lines — checkbox + label pattern)
├── tabs/
│   ├── ExplorerTab.tsx        (~400 lines — data grid, column filters, field panel)
│   ├── AccuracyTab.tsx        (~350 lines — KPIs, charts, lag curves, champion panel)
│   ├── ClustersTab.tsx        (~150 lines — cluster summary, profiles, visualization)
│   ├── DfuAnalysisTab.tsx     (~250 lines — sales vs forecast overlay chart)
│   ├── MarketIntelTab.tsx     (~120 lines — search + narrative briefing)
│   └── ChatPanel.tsx          (~120 lines — chat drawer/panel)
├── types/
│   └── index.ts              (~80 lines — all 18 type definitions)
```

### Key Actions

1. **Create `api/client.ts`** — Centralized fetch wrapper replacing 19 copy-pasted try/catch/cancelled blocks
2. **Extract reusable components** — TabButton (5 copies → 1), KpiCard (15+ → 1), LoadingOverlay (3 → 1)
3. **Extract tab components** — One file per tab with its own local state
4. **Create custom hooks** — `useFetch<T>()` generic hook eliminates the repeated useEffect pattern; `useExplorer()`, `useAccuracy()`, `useDfuAnalysis()` encapsulate per-tab state
5. **Move types** to `types/index.ts`

### Result

| Metric | Before | After |
|--------|--------|-------|
| App.tsx lines | 2,814 | ~120 |
| Total frontend lines | 2,814 | ~2,100 across 18 files |
| useState in largest component | 40 | ~8 |
| Duplicated fetch pattern | 19 copies | 1 shared hook |
| Tab button copies | 5 | 1 component |
| Testability | Impossible | Per-tab component tests |

---

## R4. Error Handling & Observability

**Priority: MEDIUM | Effort: LOW | Risk: LOW**

### Problem

- **Zero logging** — no `import logging` anywhere in `api/main.py`; all scripts use bare `print()`
- **5 silent exception catches** swallowing failures:
  - Line 2429: Google API search — `except Exception: pass`
  - Line 2477: OpenAI responses API — `except Exception: pass`
  - Line 2564: pgvector search — `except Exception: return []`
- **No request middleware** — no request IDs, latency tracking, or status code logging
- **No pool shutdown** — connection pool lazily created but never cleaned up

### Key Actions

1. **Add structured logging** — Python `logging` module in API; replace `print()` in scripts
2. **Add FastAPI lifespan** — Proper pool startup/shutdown
3. **Replace silent catches** — Log warnings instead of swallowing exceptions
4. **Add global exception handler** — Consistent error responses with `{"error": str, "detail": str}`
5. **Add CORS middleware** — Required for frontend at `:5173` to communicate with API at `:8000`

---

## R5. Database Access Improvements

**Priority: MEDIUM | Effort: MEDIUM | Risk: MEDIUM**

### Problem

- **23 inline `with get_conn()`** blocks with raw SQL across endpoints
- **7 f-string SQL patterns** for table/column names (safe via `qident()`, but fragile)
- **Unguarded `fetchone()[0]`** — will crash if query returns no rows (lines 322, 419, 1356)
- **No connection lifecycle** — pool created lazily, never closed

### Key Actions

1. **Add null guards** — Check `fetchone()` return before indexing
2. **Create query helpers** for repeated patterns:
   - `fetch_page()` already exists — route all 7 domain page endpoints through it
   - Extract shared accuracy filter builder (used by both slice and lag-curve)
3. **Add pool lifecycle** via FastAPI `lifespan` context manager
4. **Parameterize all user-facing values** — ensure no model_id or filter values enter SQL via f-string

---

## R6. Shared Utilities — Consolidate `common/`

**Priority: MEDIUM | Effort: LOW | Risk: LOW**
**Bundle with R1 for maximum impact**

### Problem

`common/` contains only `domain_specs.py` (429 lines). Meanwhile, identical utility code is duplicated across 6+ scripts.

### Duplication Map

| Function | Duplicated In | Copies |
|----------|---------------|--------|
| `get_db_conn()` | 6 scripts (all backtests + load + champion) | 6 |
| `_ts()` timestamp helper | 5 scripts (4 backtests + champion) | 5 |
| WAPE/accuracy/bias formulas | 4 backtest scripts + API `_compute_kpis()` | 5 |
| `generate_timeframes()` | 4 backtest scripts | 4 |
| `assign_execution_lag()` | 4 backtest scripts | 4 |
| `expand_to_all_lags()` | 4 backtest scripts | 4 |
| Output column definitions | 4 backtest scripts | 4 |

### New Modules

| Module | Lines | Source of Truth For |
|--------|-------|---------------------|
| `common/db.py` | ~15 | Database connection params |
| `common/metrics.py` | ~30 | WAPE, accuracy, bias (shared by API + scripts) |
| `common/constants.py` | ~20 | LAG_RANGE, ROLLING_WINDOWS, output columns, model IDs |
| `common/feature_engineering.py` | ~120 | Feature matrix, lag features, forecast_ck |
| `common/backtest_framework.py` | ~400 | Orchestrator base class |
| `common/mlflow_utils.py` | ~50 | Experiment logging wrapper |

---

## R7. Infrastructure & Minor Fixes

**Priority: LOW | Effort: LOW | Risk: NONE**

| # | Issue | Location | Fix |
|---|-------|----------|-----|
| 1 | **Vite config bug** — `vite.config.ts` has duplicate proxy entries (`/dfu`, `/competition` appear twice, lines 31-37 AND 47-54) | `frontend/vite.config.ts` | Remove duplicate proxy block |
| 2 | **Duplicate vite configs** — both `.js` and `.ts` versions exist | `frontend/` | Remove stale `vite.config.js`, keep `.ts` (after fixing #1) |
| 3 | **SQL 012 missing from Makefile** — `012_create_dfu_coverage_view.sql` not in `db-apply-sql` target | `Makefile` ~line 100 | Add to the `db-apply-sql` chain |
| 4 | **No `spark-all` target** — 7 individual spark targets exist but no composite | `Makefile` | Add `spark-all` composite target |
| 5 | **`trino_queries.sql` unused** — reference queries not integrated into any target | `sql/` | Integrate into `trino-check` or document as reference |
| 6 | **Magic strings** — model IDs like `"external"`, `"champion"`, `"ceiling"` scattered | API + scripts | Centralize in `common/constants.py` |
| 7 | **Makefile target duplication** — Spark targets repeat 11 lines of Docker env setup 7 times | `Makefile` lines 188-277 | Extract `SPARK_ENV` variable or define function |

---

## Priority & Impact Matrix

| # | Recommendation | Effort | Lines Saved | Risk | Do When |
|---|---------------|--------|-------------|------|---------|
| **R1** | Backtest framework extraction | Medium | **~2,400** | Low | **Phase 1** — isolated, no UI impact |
| **R6** | Common utilities consolidation | Low | **~200** | Low | **Phase 1** — bundle with R1 |
| **R2** | API router decomposition | Medium | **~900** | Medium | **Phase 2** — test endpoints after |
| **R4** | Error handling & observability | Low | 0 (net add) | Low | **Phase 2** — bundle with R2 |
| **R5** | DB access improvements | Medium | **~200** | Medium | **Phase 2** — bundle with R2 |
| **R3** | Frontend decomposition | High | **~700** | Medium | **Phase 3** — visual regression risk |
| **R7** | Infrastructure & minor fixes | Low | **~50** | None | **Anytime** |

### Execution Order Rationale

1. **Phase 1 (R1 + R6):** Backtest scripts are completely isolated from the running application. Refactoring them has zero risk of breaking the API or UI. This phase delivers the largest line reduction (~2,600 lines) with the lowest risk.

2. **Phase 2 (R2 + R4 + R5):** API decomposition requires endpoint testing but no frontend changes. Adding proper error handling and connection lifecycle management improves reliability.

3. **Phase 3 (R3):** Frontend decomposition carries the highest visual regression risk and requires careful testing across all 5 tabs. Do this last when the backend is stable.

4. **Phase 4 (R7):** Minor fixes can be done anytime as independent patches.

---

## Expected Outcome

| Metric | Before | After |
|--------|--------|-------|
| Largest single file | 2,814 lines (`App.tsx`) | ~400 lines (`ExplorerTab.tsx`) |
| Files exceeding 500 lines | 6 | 0 |
| Duplicated code | ~3,800 lines | ~200 lines |
| Total codebase | ~12,700 lines | ~8,900 lines |
| Time to add new ML model | Copy 830 lines, edit 3 places | Write ~130 lines implementing adapter |
| Time to add new UI tab | Edit 2,814-line file, add 40th+ useState | Create 1 new file (~150 lines) |
| Time to fix accuracy formula | Edit 5 files (4 scripts + API) | Edit 1 file (`common/metrics.py`) |
| Backend testability | Untestable (2,706-line god file) | Per-router unit tests |
| Frontend testability | Untestable (monolithic component) | Per-tab component tests |

---

## Verification Plan

After each refactoring phase, validate:

1. **`make check-all`** — DB connectivity, API health, Trino (if configured)
2. **`make api` + `make ui`** — Frontend renders all 5 tabs correctly
3. **`make backtest-lgbm`** — Backtest pipeline produces identical CSV output
4. **`make backtest-catboost`** / **`make backtest-xgboost`** — All models still work
5. **`make champion-select`** — Champion selection runs without errors
6. **Manual smoke test:** Explorer pagination + filtering, Accuracy KPIs + charts, DFU analysis chart with model toggles, Market Intelligence search, Chat NL→SQL
7. **Diff backtest outputs** — Compare CSV outputs before/after to confirm identical predictions

---

## Appendix: File-Level Refactoring Map

### Files to Create (Phase 1)
- `common/db.py` — Replaces 6 copies of `get_db_conn()`
- `common/constants.py` — Replaces 4 copies of LAG_RANGE, ROLLING_WINDOWS, output columns
- `common/metrics.py` — Replaces 5 copies of WAPE/accuracy/bias
- `common/feature_engineering.py` — Replaces 3 copies of `build_feature_matrix()`, `mask_future_sales()`
- `common/backtest_framework.py` — Replaces ~400 lines duplicated across 4 scripts
- `common/mlflow_utils.py` — Replaces 4 copies of MLflow logging

### Files to Create (Phase 2)
- `api/deps.py` — Connection pool lifecycle
- `api/models.py` — Pydantic response schemas
- `api/routers/domains.py` — Generic domain CRUD (replaces 14 alias endpoints)
- `api/routers/accuracy.py` — Accuracy slice + lag-curve (shared filter logic)
- `api/routers/competition.py` — Champion model selection
- `api/routers/dfu_analysis.py` — DFU analysis + clusters
- `api/routers/chat.py` — NL→SQL chatbot
- `api/routers/market_intel.py` — Market intelligence
- `api/routers/bench.py` — Benchmark endpoints
- `api/services/kpi.py` — KPI computation (imports from `common/metrics.py`)
- `api/services/filters.py` — Shared filter/WHERE builders

### Files to Create (Phase 3)
- `frontend/src/api/client.ts` — Fetch wrapper
- `frontend/src/hooks/useFetch.ts` — Generic data hook
- `frontend/src/hooks/useExplorer.ts` — Explorer state
- `frontend/src/hooks/useAccuracy.ts` — Accuracy state
- `frontend/src/hooks/useDfuAnalysis.ts` — DFU analysis state
- `frontend/src/hooks/useDebounce.ts` — Debounce hook (extract from App.tsx)
- `frontend/src/components/TabButton.tsx` — Tab navigation button
- `frontend/src/components/KpiCard.tsx` — KPI display card
- `frontend/src/components/LoadingOverlay.tsx` — Loading animation
- `frontend/src/components/ModelWinsBar.tsx` — Model wins visualization
- `frontend/src/components/FilterInput.tsx` — Input + datalist
- `frontend/src/tabs/ExplorerTab.tsx` — Data explorer
- `frontend/src/tabs/AccuracyTab.tsx` — Accuracy analytics
- `frontend/src/tabs/ClustersTab.tsx` — Cluster view
- `frontend/src/tabs/DfuAnalysisTab.tsx` — DFU analysis
- `frontend/src/tabs/MarketIntelTab.tsx` — Market intelligence
- `frontend/src/tabs/ChatPanel.tsx` — Chat interface
- `frontend/src/types/index.ts` — Shared TypeScript types

### Files to Modify Significantly
- `scripts/run_backtest.py` — 846 → ~130 lines (LGBM adapter only)
- `scripts/run_backtest_catboost.py` — 836 → ~130 lines (CatBoost adapter only)
- `scripts/run_backtest_xgboost.py` — 831 → ~130 lines (XGBoost adapter only)
- `scripts/run_backtest_prophet.py` — 751 → ~200 lines (Prophet adapter only)
- `api/main.py` — 2,706 → ~60 lines (app factory only)
- `frontend/src/App.tsx` — 2,814 → ~120 lines (layout shell only)

### Files to Fix (Phase 4)
- `frontend/vite.config.ts` — Remove duplicate proxy entries
- `frontend/vite.config.js` — Delete (stale duplicate of .ts)
- `Makefile` — Add SQL 012 to db-apply-sql, add spark-all target
