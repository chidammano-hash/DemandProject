# Feature 31 — Comprehensive Testing Strategy

> **Status:** Spec Complete — Ready for Implementation
> **Priority:** Critical
> **Scope:** Backend (Python/FastAPI) + Frontend (React/TypeScript) + Integration + Performance
> **Prerequisite for:** All future feature development

---

## 1. Motivation

Demand Studio has 40 API endpoints, 6 frontend tabs, 8 backtesting model families, a clustering pipeline, champion model selection, and a shared backtest framework totaling ~15,000+ lines of business logic. Current test coverage is minimal — **1 frontend test file** (formatters only) and **0 backend tests**. As the platform grows, regressions, data integrity bugs, and accuracy calculation errors become increasingly costly to diagnose. A comprehensive testing strategy is essential to ensure correctness, prevent regressions, and enable confident refactoring.

---

## 2. Testing Philosophy

### 2.1 Testing Pyramid

```
         ╱╲
        ╱ E2E ╲           ← Few, expensive, high confidence
       ╱────────╲
      ╱Integration╲       ← Moderate count, API + DB + UI integration
     ╱──────────────╲
    ╱   Unit Tests    ╲    ← Many, fast, isolated
   ╱────────────────────╲
```

**Target distribution:**
- **Unit tests:** 70% — pure functions, calculations, transformations, hooks, components
- **Integration tests:** 25% — API endpoints with DB, frontend with mocked API, pipeline stages
- **E2E tests:** 5% — critical user journeys (data exploration, accuracy review, DFU analysis)

### 2.2 Core Principles

1. **Test behavior, not implementation** — assert on outputs and side effects, not internal state
2. **Fast feedback** — unit tests must run in < 30 seconds; integration tests in < 2 minutes
3. **Deterministic** — no flaky tests; mock external services (OpenAI, Google, MLflow)
4. **Co-located** — test files live next to source files or in `__tests__/` directories
5. **Mandatory for new features** — every PR must include tests for new/changed behavior (see Section 12)

---

## 3. Backend Testing (Python / pytest)

### 3.1 Test Infrastructure

| Component | Technology |
|---|---|
| Framework | pytest >= 8.3 |
| Async support | pytest-asyncio |
| HTTP testing | httpx (ASGI transport for FastAPI) |
| DB fixtures | pytest fixtures with transaction rollback |
| Mocking | unittest.mock / pytest-mock |
| Coverage | pytest-cov |
| Factories | Custom fixtures (no factory_boy needed at this scale) |

**Directory structure:**
```
mvp/demand/
├── tests/
│   ├── conftest.py              # Shared fixtures (DB connection, test client, sample data)
│   ├── unit/
│   │   ├── test_metrics.py
│   │   ├── test_feature_engineering.py
│   │   ├── test_constants.py
│   │   ├── test_domain_specs.py
│   │   ├── test_backtest_framework.py
│   │   └── test_mlflow_utils.py
│   ├── api/
│   │   ├── test_health.py
│   │   ├── test_domains.py
│   │   ├── test_forecast_accuracy.py
│   │   ├── test_dfu_analysis.py
│   │   ├── test_competition.py
│   │   ├── test_clusters.py
│   │   ├── test_benchmarking.py
│   │   ├── test_chat.py
│   │   └── test_market_intel.py
│   ├── scripts/
│   │   ├── test_normalize.py
│   │   ├── test_load_postgres.py
│   │   ├── test_clustering_pipeline.py
│   │   ├── test_champion_selection.py
│   │   └── test_backtest_load.py
│   └── integration/
│       ├── test_data_pipeline.py
│       └── test_backtest_pipeline.py
```

**Make targets to add:**
```makefile
test:              ## Run all Python tests
	cd $(DIR) && uv run pytest tests/ -v --tb=short

test-unit:         ## Run unit tests only
	cd $(DIR) && uv run pytest tests/unit/ -v --tb=short

test-api:          ## Run API tests only
	cd $(DIR) && uv run pytest tests/api/ -v --tb=short

test-integration:  ## Run integration tests (requires running DB)
	cd $(DIR) && uv run pytest tests/integration/ -v --tb=short

test-cov:          ## Run tests with coverage report
	cd $(DIR) && uv run pytest tests/ --cov=api --cov=common --cov=scripts --cov-report=term-missing

test-all:          ## Run all tests (Python + frontend)
	$(MAKE) test && $(MAKE) ui-test
```

### 3.2 Shared Fixtures (`conftest.py`)

```python
# Key fixtures needed:

@pytest.fixture(scope="session")
def db_connection():
    """Real Postgres connection for integration tests. Rolls back after session."""

@pytest.fixture
def test_client():
    """httpx.AsyncClient bound to FastAPI app via ASGI transport."""

@pytest.fixture
def sample_sales_df():
    """Small DataFrame mimicking fact_sales_monthly (10 rows, 3 items, 2 locations)."""

@pytest.fixture
def sample_forecast_df():
    """Small DataFrame mimicking fact_external_forecast_monthly (multiple models, lags 0-4)."""

@pytest.fixture
def sample_dfu_df():
    """Small DataFrame mimicking dim_dfu with cluster assignments."""

@pytest.fixture
def mock_db_pool():
    """Mocked psycopg connection pool for unit testing API endpoints."""
```

### 3.3 Unit Tests — Common Modules

#### 3.3.1 `test_metrics.py` — `common/metrics.py`

**Target:** `compute_accuracy_metrics(forecast_col, actual_col)`

| Test Case | Input | Expected Output |
|---|---|---|
| Perfect forecast | f=[100,200], a=[100,200] | wape=0, bias=0, accuracy=100 |
| 50% over-forecast | f=[150,300], a=[100,200] | wape=50, bias=0.5, accuracy=50 |
| 50% under-forecast | f=[50,100], a=[100,200] | wape=50, bias=-0.5, accuracy=50 |
| All zeros actual | f=[100], a=[0] | wape=None, bias=None, accuracy=None |
| Empty series | f=[], a=[] | n_rows=0, all None |
| NaN in data | f=[100,NaN], a=[100,200] | drops NaN row, computes on remainder |
| Single row | f=[120], a=[100] | wape=20, bias=0.2, accuracy=80 |
| Negative actuals | f=[100], a=[-100] | handles abs(total_a) correctly |
| Mixed positive/negative actuals | f=[100,100], a=[200,-50] | correct sum-based calculation |
| Large numbers (overflow check) | f=[1e12], a=[1e12] | wape=0, no overflow |

#### 3.3.2 `test_feature_engineering.py` — `common/feature_engineering.py`

**Target:** `build_feature_matrix()`

| Test Case | Description |
|---|---|
| Lag features correctness | qty_lag_1 through qty_lag_12 match shifted values |
| Rolling window stats | 3/6/12-month rolling mean/std/min/max are correct |
| Calendar features | month_of_year, quarter, is_holiday populated correctly |
| No future leakage | For each target month, only past data used |
| Missing months | Gaps in time series handled (NaN lags, not shifted wrong) |
| Single DFU | Feature matrix built correctly for 1 DFU |
| Multiple DFUs | Features don't leak across DFU boundaries |
| Categorical features | `cat_dtype` parameter correctly types categorical columns |
| Minimum history threshold | DFUs with < MIN_TRAIN_MONTHS excluded |

#### 3.3.3 `test_backtest_framework.py` — `common/backtest_framework.py`

**Target:** Core orchestrator functions

| Function | Test Cases |
|---|---|
| `generate_timeframes()` | Returns 10 timeframes A–J; each train_end < test_start; test windows are 1 month; expanding windows grow monotonically |
| `assign_execution_lag()` | Lag 0 = same month; lag 1 = 1 month ahead; lag 4 = 4 months ahead; out-of-range months excluded |
| `expand_to_all_lags()` | Single prediction expands to 5 rows (lag 0–4); columns match `ARCHIVE_COLS` |
| `postprocess_predictions()` | Sorts by forecast_ck + month; removes duplicates; validates required columns exist |
| `load_backtest_data()` | Filters to TYPE=1 sales; joins DFU attributes; date parsing correct |
| `save_backtest_output()` | Creates output directory; writes CSV + JSON metadata; metadata contains model_id, timeframes, row counts |

#### 3.3.4 `test_domain_specs.py` — `common/domain_specs.py`

| Test Case | Description |
|---|---|
| All 7 domains defined | item, location, customer, time, dfu, sales, forecast present |
| Column/type alignment | Every column has a matching type entry |
| Search fields subset | search_fields is a subset of columns |
| Business key validity | ck_columns exist in columns list |
| No duplicate columns | No repeated column names within a domain |

#### 3.3.5 `test_constants.py` — `common/constants.py`

| Test Case | Description |
|---|---|
| LAG_RANGE bounds | 1 ≤ min, max ≤ 12 |
| ROLLING_WINDOWS sorted | [3, 6, 12] ascending |
| Output column lists | FACT_COLS and ARCHIVE_COLS are non-empty, no duplicates |
| CAT_FEATURES validity | All category features are strings, non-empty |

#### 3.3.6 `test_mlflow_utils.py` — `common/mlflow_utils.py`

| Test Case | Description |
|---|---|
| Logs correct experiment | MLflow experiment name matches model family |
| Params logged | model_type, hyperparams, lag included |
| Metrics logged | wape, accuracy_pct, bias recorded |
| No MLflow server | Graceful handling when MLflow unavailable (mock) |

---

### 3.4 API Endpoint Tests

All API tests use `httpx.AsyncClient` with `ASGITransport(app=app)`. DB interactions are mocked or use a test database with transaction rollback.

#### 3.4.1 `test_health.py`

| Endpoint | Test Case |
|---|---|
| `GET /health` | Returns 200 with status "ok" |
| `GET /` | Redirects to docs or returns root response |

#### 3.4.2 `test_domains.py`

| Endpoint | Test Cases |
|---|---|
| `GET /domains` | Returns all 7 domain names |
| `GET /domains/{domain}/meta` | Returns columns, types, search fields for each domain |
| `GET /domains/{domain}/meta` | Returns 404 for invalid domain |
| `GET /domains/{domain}/page` | Default pagination (limit=50, offset=0) |
| `GET /domains/{domain}/page` | Respects limit param (50–1000 range) |
| `GET /domains/{domain}/page` | Rejects limit < 50 or > 1000 |
| `GET /domains/{domain}/page` | Offset pagination works (offset=100) |
| `GET /domains/{domain}/page?sort=column&order=desc` | Sort + order applied |
| `GET /domains/{domain}/page?search=text` | GIN trigram search returns filtered rows |
| `GET /domains/{domain}/page?column==exact` | B-tree exact match (= prefix) |
| `GET /domains/{domain}/page?column=substring` | Trigram substring match |
| `GET /domains/{domain}/suggest?field=X&prefix=abc` | Returns matching suggestions |
| `GET /domains/{domain}/suggest` | Returns 400 for missing field param |
| `GET /domains/{domain}/sample-pair` | Returns item + location pair |

#### 3.4.3 `test_forecast_accuracy.py`

| Endpoint | Test Cases |
|---|---|
| `GET /forecast/accuracy/slice?group_by=cluster` | Returns accuracy grouped by cluster |
| `GET /forecast/accuracy/slice?group_by=supplier` | Returns accuracy grouped by supplier |
| `GET /forecast/accuracy/slice?models=lgbm,external` | Filters to specified models |
| `GET /forecast/accuracy/slice?window=6` | 6-month rolling window |
| `GET /forecast/accuracy/slice` | Default window (12 months) |
| `GET /forecast/accuracy/lag-curve` | Returns lag 0–4 accuracy by model |
| `GET /forecast/accuracy/lag-curve?models=external` | Filtered to single model |
| `GET /domains/forecast/models` | Returns list of all model IDs in database |

**Accuracy calculation validation:**
- Known dataset with hand-calculated WAPE, bias, accuracy
- Verify formula: `accuracy = 100 - (100 * SUM(ABS(F-A)) / ABS(SUM(A)))`
- Verify bias: `(SUM(F) / SUM(A)) - 1`
- Edge case: all actuals zero → null metrics
- Edge case: single DFU single month

#### 3.4.4 `test_dfu_analysis.py`

| Endpoint | Test Cases |
|---|---|
| `GET /dfu/analysis?item=X&location=Y` | Returns sales + forecast overlay for item@location |
| `GET /dfu/analysis?item=X&location=Y&mode=all_items_at_location` | Aggregated across items |
| `GET /dfu/analysis?item=X&location=Y&mode=item_at_all_locations` | Aggregated across locations |
| `GET /dfu/analysis` | Returns 422 for missing required params |
| Response shape | Contains: mode, item, location, points, models[], series[], model_monthly{}, dfu_attributes[] |
| Multi-model | Multiple model series returned when data exists |
| No data | Empty series when item/location combo has no data |

#### 3.4.5 `test_competition.py`

| Endpoint | Test Cases |
|---|---|
| `GET /competition/config` | Returns current config (models, metric, lag) |
| `PUT /competition/config` | Updates config; returns updated |
| `PUT /competition/config` | Rejects empty model list |
| `POST /competition/run` | Triggers champion selection (mock heavy computation) |
| `GET /competition/summary` | Returns per-DFU winners, model win counts, FVA stats |
| Champion rows | model_id='champion' rows inserted correctly |
| Ceiling oracle | model_id='ceiling' picks best per DFU per month |

#### 3.4.6 `test_clusters.py`

| Endpoint | Test Cases |
|---|---|
| `GET /domains/dfu/clusters` | Returns cluster list with count/pct/avg_demand |
| `GET /domains/dfu/clusters/profiles` | Returns profiles with optimal_k, silhouette_score |
| `GET /domains/dfu/clusters/visualization/{img}` | Returns image or 404 |
| Cluster labels | Labels match expected pattern (e.g., high_volume_steady) |

#### 3.4.7 `test_chat.py`

| Endpoint | Test Cases |
|---|---|
| `POST /chat` | Returns SQL + result for simple question (mock OpenAI) |
| `POST /chat` | Read-only enforcement (rejects INSERT/UPDATE/DELETE) |
| `POST /chat` | 5-second timeout enforced |
| `POST /chat` | 500-row limit enforced |
| `POST /chat` | Returns error for malformed questions |
| Missing API key | Returns 503 when OPENAI_API_KEY not set |

#### 3.4.8 `test_market_intel.py`

| Endpoint | Test Cases |
|---|---|
| `POST /market-intelligence` | Returns search results + narrative (mock Google + OpenAI) |
| `POST /market-intelligence` | Looks up item metadata (description, brand, category) |
| `POST /market-intelligence` | Looks up location state |
| Missing API keys | Returns 503 when GOOGLE_API_KEY/CSE_ID not set |
| Invalid item/location | Returns 404 or empty results |

#### 3.4.9 `test_benchmarking.py`

| Endpoint | Test Cases |
|---|---|
| `GET /bench/compare?domain=sales` | Returns Postgres + Trino latencies per query type |
| Response shape | Contains per-query min/max/avg/p50/p95 + winner + speedup |
| Trino unavailable | Graceful error when Trino is down |

---

### 3.5 Script Tests

#### 3.5.1 `test_normalize.py` — `scripts/normalize_dataset_csv.py`

| Test Case | Description |
|---|---|
| Null normalization | `''`, `'null'`, `'none'`, `'NA'` → NULL |
| Type casting (int) | String "123" → int 123; "abc" → NULL |
| Type casting (float) | String "12.5" → float 12.5; "N/A" → NULL |
| Type casting (date) | Various date formats → ISO date; invalid → NULL |
| Sales TYPE filter | Only TYPE=1 rows retained for sales domain |
| Lag computation | `month_diff` calculated correctly for forecast domain |
| Column ordering | Output columns match domain spec |
| Idempotency | Running normalize twice produces identical output |
| Empty input CSV | Handles gracefully (empty output or error) |
| Unicode characters | Non-ASCII item descriptions preserved |

#### 3.5.2 `test_load_postgres.py` — `scripts/load_dataset_postgres.py`

| Test Case | Description |
|---|---|
| Correct table targeted | Each domain loads to correct table |
| Row count matches | Loaded rows = input CSV rows (minus header) |
| Surrogate keys | sk auto-generated, unique |
| Timestamps | load_ts and modified_ts populated |
| Duplicate handling | UPSERT on business key (no duplicate ck) |
| Materialized view refresh | Views refreshed after load |

#### 3.5.3 `test_clustering_pipeline.py`

| Test Case | Description |
|---|---|
| Feature generation | Output matrix has expected columns (time series + item + DFU features) |
| No NaN in features | Feature matrix cleaned of NaN (imputed or excluded) |
| KMeans convergence | Model trains without error on sample data |
| Optimal K selection | K selected within configured range |
| Label assignment | Each cluster gets a label from config vocabulary |
| DB update | cluster_assignment written to dim_dfu correctly |
| MLflow logging | Experiment logged with K, silhouette, inertia |

#### 3.5.4 `test_champion_selection.py`

| Test Case | Description |
|---|---|
| Single model | Champion = only model available |
| Multiple models, clear winner | Lowest WAPE model wins per DFU |
| Tie-breaking | Consistent winner when WAPE identical |
| Ceiling oracle | Picks best model per DFU per month |
| Gap to ceiling | gap = champion_wape - ceiling_wape ≥ 0 |
| Config loading | Reads model_competition.yaml correctly |
| Missing model data | DFU skipped if a model has no predictions |

#### 3.5.5 `test_backtest_load.py` — `scripts/load_backtest_forecasts.py`

| Test Case | Description |
|---|---|
| Main table load | Predictions inserted into fact_external_forecast_monthly |
| Archive table load | All-lags rows inserted into backtest_lag_archive |
| Duplicate prevention | UNIQUE(forecast_ck, model_id) enforced |
| View refresh | 5 materialized views refreshed after load |
| model_id preserved | model_id from CSV matches DB rows |

---

### 3.6 Integration Tests

#### 3.6.1 `test_data_pipeline.py`

**Requires:** Running PostgreSQL instance (Docker)

| Test Case | Description |
|---|---|
| CSV → normalize → load → query | End-to-end for a small synthetic dataset |
| Data integrity | Loaded data matches source after round-trip |
| Materialized view consistency | Aggregated views match raw data sums |
| Index usage | EXPLAIN shows GIN/B-tree index scans for filtered queries |

#### 3.6.2 `test_backtest_pipeline.py`

**Requires:** Running PostgreSQL + sample sales/forecast data loaded

| Test Case | Description |
|---|---|
| LGBM backtest mini-run | 2 timeframes, 5 DFUs, predictions generated |
| Predictions shape | Output has required columns, no NaN in forecast values |
| Lag expansion | Each prediction expanded to lags 0–4 |
| Load + query | Loaded predictions queryable via accuracy endpoint |
| Multi-model | Two models backtested, both queryable |

---

## 4. Frontend Testing (React / Vitest)

### 4.1 Test Infrastructure

| Component | Technology |
|---|---|
| Framework | Vitest 4.x |
| Component testing | React Testing Library |
| DOM environment | jsdom |
| Assertions | @testing-library/jest-dom |
| API mocking | MSW (Mock Service Worker) or vitest mocking |
| User events | @testing-library/user-event |
| Query testing | @tanstack/react-query test utils |

**Directory structure:**
```
mvp/demand/frontend/src/
├── __tests__/
│   ├── setup.ts                    # (existing) test setup
│   └── formatters.test.ts          # (existing) formatter tests
├── api/
│   └── __tests__/
│       └── queries.test.ts         # TanStack Query layer tests
├── hooks/
│   ├── __tests__/
│   │   ├── useTheme.test.ts
│   │   ├── useUrlState.test.ts
│   │   ├── useKeyboardShortcuts.test.ts
│   │   └── useDebounce.test.ts
├── components/
│   ├── __tests__/
│   │   ├── DataTable.test.tsx
│   │   ├── Skeleton.test.tsx
│   │   ├── EChartContainer.test.tsx
│   │   └── KeyboardShortcutHelp.test.tsx
├── tabs/
│   ├── __tests__/
│   │   ├── ExplorerTab.test.tsx
│   │   ├── AccuracyTab.test.tsx
│   │   ├── DfuAnalysisTab.test.tsx
│   │   ├── ClustersTab.test.tsx
│   │   ├── MarketIntelTab.test.tsx
│   │   └── ChatPanel.test.tsx
├── lib/
│   ├── __tests__/
│   │   ├── export.test.ts
│   │   └── formatters.test.ts      # (move existing here)
```

### 4.2 Unit Tests — Utility Functions

#### 4.2.1 `formatters.test.ts` (existing — extend)

**Already covered:** formatNumber, formatPercent, formatCell, formatCompactNumber, titleCase

**Add these test cases:**

| Function | Additional Test Cases |
|---|---|
| `formatNumber` | Negative numbers; very large numbers (1e15); decimal precision |
| `formatPercent` | 0%; 100%; > 100%; negative % |
| `formatCompactNumber` | 0; negative numbers; decimals; billions (B suffix) |
| `formatCell` | Date strings; boolean values; objects |

#### 4.2.2 `export.test.ts` — `lib/export.ts`

| Test Case | Description |
|---|---|
| CSV generation | Correct header row + data rows |
| Special characters | Commas, quotes, newlines in values escaped |
| Empty data | Produces header-only CSV |
| Column ordering | Matches provided column order |
| Null handling | Null/undefined → empty cell |
| Download trigger | Blob URL created and link clicked (mock) |

### 4.3 Unit Tests — Custom Hooks

#### 4.3.1 `useTheme.test.ts`

| Test Case | Description |
|---|---|
| Default theme | Returns "light" when no localStorage |
| Persists to localStorage | Setting theme writes to localStorage |
| Reads from localStorage | Initializes with stored theme |
| Applies CSS class | document.documentElement gets correct class |
| Toggle cycles | light → dark → midnight → light |
| Invalid stored value | Falls back to "light" |

#### 4.3.2 `useUrlState.test.ts`

| Test Case | Description |
|---|---|
| Reads from URL | Initial state from query params |
| Writes to URL | State change updates query params |
| Default values | Missing params use defaults |
| Multiple params | Multiple state values synced independently |
| Browser back/forward | State updates on popstate |
| Encoding | Special characters in values URL-encoded |

#### 4.3.3 `useKeyboardShortcuts.test.ts`

| Test Case | Description |
|---|---|
| Tab switching | Keys 1–5 switch tabs |
| Search focus | `/` focuses search input |
| Escape closes | Esc closes modals/panels |
| Help toggle | `?` toggles shortcut help |
| Ctrl+E toggle | Toggles field expansion |
| Input suppression | Shortcuts suppressed when typing in input/textarea |
| No duplicate listeners | Multiple mounts don't stack listeners |

#### 4.3.4 `useDebounce.test.ts`

| Test Case | Description |
|---|---|
| Delays value | Value updates after delay |
| Cancels on rapid change | Only last value emitted |
| Immediate first value | Initial value available immediately |
| Custom delay | Respects custom delay parameter |

### 4.4 Component Tests

#### 4.4.1 `DataTable.test.tsx`

| Test Case | Description |
|---|---|
| Renders header | Column headers match provided columns |
| Renders rows | Correct number of data rows rendered |
| Empty state | Shows "No data" message when rows=[] |
| Column sorting | Click header → sorts ascending; click again → descending |
| Column resize | Drag column border changes width |
| Row selection | Click row → row highlighted |
| CSV export | Export button triggers download |
| Pagination controls | Next/prev buttons work; page number displayed |
| Virtualization | Only visible rows rendered (check DOM node count) |
| Column visibility | Hidden columns not rendered |
| Filter input | Type in filter → rows filtered |
| Numeric column formatting | Numbers formatted with commas |
| Null cell display | Null values show dash |

#### 4.4.2 `Skeleton.test.tsx`

| Test Case | Description |
|---|---|
| Renders placeholder | Skeleton element rendered with animation class |
| Custom dimensions | Width/height props applied |
| Accessibility | Has aria-busy attribute |

#### 4.4.3 `EChartContainer.test.tsx`

| Test Case | Description |
|---|---|
| Renders container div | Chart container div present in DOM |
| Theme awareness | Dark theme → dark chart options |
| Cleanup | Chart instance disposed on unmount |
| Resize handling | Resizes chart on window resize |

#### 4.4.4 `KeyboardShortcutHelp.test.tsx`

| Test Case | Description |
|---|---|
| Renders when open | Shows shortcut list when open=true |
| Hidden when closed | Not in DOM when open=false |
| Lists all shortcuts | 1-5, /, Esc, ?, Ctrl+E all listed |
| Close button | Close button calls onClose |
| Escape closes | Pressing Esc calls onClose |

### 4.5 Tab Component Tests

Each tab test uses mocked API responses via MSW or vitest mocks. Tests verify rendering, loading states, error states, and user interactions.

#### 4.5.1 `ExplorerTab.test.tsx`

| Test Case | Description |
|---|---|
| Domain selector | Dropdown lists all 7 domains |
| Domain switch | Selecting domain fetches new data |
| Data grid renders | Table rows appear after API response |
| Loading state | Skeleton/loading shown while fetching |
| Error state | Error boundary message on API failure |
| Search | Typing in search bar filters results |
| Filter by column | Column filter input triggers filtered request |
| Exact match filter | `=prefix` triggers B-tree exact match |
| Pagination | Next/prev loads new page |
| Page size | Changing limit re-fetches |
| Sort | Column header click sorts data |
| Column typeahead | Typing in column filter shows suggestions |
| CSV export | Export button downloads current view |
| Approximate count | Badge shows "100,000+" for large sets |
| Column visibility | Toggle columns on/off |

#### 4.5.2 `AccuracyTab.test.tsx`

| Test Case | Description |
|---|---|
| KPI cards render | Accuracy %, WAPE, Bias, Total Forecast, Total Actual displayed |
| Model selector | Lists available models |
| Window selector | 1–12 month windows |
| Accuracy by dimension | Group by cluster/supplier renders table |
| Lag curve chart | Chart renders with lag 0–4 data |
| Champion KPIs | Champion + ceiling cards shown when data exists |
| Model wins chart | Dual bar chart renders model win counts |
| Gap to ceiling | Gap indicator shows champion vs ceiling difference |
| Loading states | Skeletons during fetch |
| Error states | Error message on fetch failure |
| Empty state | "No data" when no models in DB |

#### 4.5.3 `DfuAnalysisTab.test.tsx`

| Test Case | Description |
|---|---|
| Item/location inputs | Text inputs for item + location |
| Sample pair button | Loads sample item/location pair |
| Mode selector | 3 modes: item@location, all_items@location, item@all_locations |
| Chart renders | Sales + forecast lines rendered |
| Multi-model overlay | Multiple model series on chart |
| Per-model KPIs | KPI cards for each model |
| Measure toggles | Toggle forecast/actual/error visibility |
| DFU attributes | Attribute panel shows item/DFU metadata |
| Loading state | Loading indicator during fetch |
| No data | "No data found" message |

#### 4.5.4 `ClustersTab.test.tsx`

| Test Case | Description |
|---|---|
| Cluster table | Lists clusters with count, pct, avg_demand |
| Cluster profiles | Profile cards with feature metrics |
| K-selection chart | Elbow/silhouette chart renders |
| Empty state | "No clusters" when none assigned |

#### 4.5.5 `MarketIntelTab.test.tsx`

| Test Case | Description |
|---|---|
| Item/location inputs | Input fields rendered |
| Submit button | Triggers API call |
| Search results | Google results listed with title, link, snippet |
| Narrative | GPT-4o narrative rendered as formatted text |
| Loading state | Loading during API call |
| Error handling | Error message on API failure |
| Missing API keys | Appropriate error for 503 |

#### 4.5.6 `ChatPanel.test.tsx`

| Test Case | Description |
|---|---|
| Input field | Chat input rendered |
| Send message | Enter key or button sends message |
| User message displayed | User message appears in chat history |
| Assistant response | SQL + result table rendered |
| Error response | Error message displayed |
| Empty response | "No results" message |
| Chat history | Multiple messages in order |
| SQL display | SQL query shown in code block |

### 4.6 API Layer Tests

#### 4.6.1 `queries.test.ts` — `api/queries.ts`

| Test Case | Description |
|---|---|
| Query key uniqueness | Each endpoint has unique query key |
| Stale time constants | FOREVER, TEN_MIN, etc. are correct durations |
| Fetch error handling | Network errors throw with message |
| Response parsing | JSON responses parsed correctly |
| URL construction | Query params correctly encoded |
| Abort signal | Requests abortable via signal |

---

## 5. Type Safety Tests

### 5.1 TypeScript Strict Mode

Ensure `tsconfig.json` has:
```json
{
  "compilerOptions": {
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitAny": true
  }
}
```

### 5.2 Pydantic Model Validation Tests

| Test Case | Description |
|---|---|
| Valid request body | Accepted without error |
| Missing required field | 422 with field name in error |
| Wrong type | 422 with type mismatch error |
| Extra fields | Ignored (or rejected per config) |
| Enum validation | Invalid enum value → 422 |

---

## 6. Data Integrity Tests

These tests verify that the data pipeline preserves correctness end-to-end.

### 6.1 Referential Integrity

| Test Case | Description |
|---|---|
| Sales → Item | Every item in fact_sales_monthly exists in dim_item |
| Sales → Location | Every location in fact_sales_monthly exists in dim_location |
| Sales → Customer | Every customer_group exists in dim_customer |
| Forecast → DFU | Every dfu_ck in forecast exists in dim_dfu |
| DFU → Item + Location | dim_dfu.item and dim_dfu.location exist in respective dims |

### 6.2 Aggregation Consistency

| Test Case | Description |
|---|---|
| Sales agg matches raw | SUM(qty) from agg_sales_monthly = SUM(qty) from fact_sales_monthly |
| Forecast agg matches raw | agg_forecast_monthly totals = fact totals |
| Champion accuracy | Champion model accuracy ≥ any individual model accuracy (per DFU) |
| Ceiling upper bound | Ceiling accuracy ≥ champion accuracy |

### 6.3 Temporal Consistency

| Test Case | Description |
|---|---|
| No future sales | max(month) in fact_sales ≤ current date |
| Lag correctness | lag = month_diff(forecast_date, actual_month) for all forecast rows |
| Timeframe ordering | Timeframe A train_end < B train_end < ... < J train_end |

---

## 7. Performance Tests

### 7.1 API Latency Benchmarks

| Endpoint | Target | Method |
|---|---|---|
| `GET /health` | < 50ms | pytest-benchmark |
| `GET /domains/{domain}/page` (no filter) | < 200ms | pytest-benchmark |
| `GET /domains/{domain}/page` (GIN search) | < 500ms | pytest-benchmark |
| `GET /forecast/accuracy/slice` | < 1s | pytest-benchmark |
| `GET /dfu/analysis` | < 2s | pytest-benchmark |

### 7.2 Frontend Render Performance

| Component | Target | Method |
|---|---|---|
| DataTable (1000 rows) | < 500ms initial render | React Profiler |
| Tab switch | < 100ms (cached) | Performance.now() |
| Theme toggle | < 50ms | No visible flicker |

### 7.3 Backend Pipeline Performance

| Operation | Target | Measurement |
|---|---|---|
| Normalize 100K rows | < 30s | time.time() |
| Load 100K rows to Postgres | < 60s | time.time() |
| Feature engineering (1000 DFUs) | < 120s | time.time() |
| KMeans training (1000 DFUs, K=5) | < 30s | time.time() |

---

## 8. Security Tests

### 8.1 SQL Injection Prevention

| Test Case | Description |
|---|---|
| Search param injection | `search='; DROP TABLE--` returns empty, no error |
| Column filter injection | `item=1 OR 1=1` treated as literal text |
| Chat read-only | INSERT/UPDATE/DELETE rejected in NL→SQL |
| Chat timeout | Long-running query killed at 5s |

### 8.2 Input Validation

| Test Case | Description |
|---|---|
| Oversized request body | > 1MB body rejected |
| Invalid domain name | `/domains/../../etc/passwd` → 404 |
| Limit bounds | limit=0 or limit=10000 → 422 |
| XSS in search | `<script>alert(1)</script>` escaped in response |

### 8.3 External API Security

| Test Case | Description |
|---|---|
| Missing API keys | Returns 503, not 500 or key leak |
| API key not in response | OPENAI_API_KEY, GOOGLE_API_KEY never in HTTP response body |

---

## 9. Accessibility Tests

| Test Case | Description |
|---|---|
| Keyboard navigation | All interactive elements reachable via Tab key |
| Focus indicators | Visible focus ring on active elements |
| Screen reader labels | aria-label on icon-only buttons |
| Color contrast | WCAG AA contrast ratios in all 3 themes |
| Loading announcements | aria-live regions for async content |
| Table headers | Data tables have proper `<th>` scope attributes |

---

## 10. Cross-Browser / Environment Tests

| Environment | Test Level |
|---|---|
| Chrome (latest) | Full E2E |
| Firefox (latest) | Smoke test (core flows) |
| Safari (latest) | Smoke test (core flows) |
| Dark mode OS preference | Theme respects prefers-color-scheme |
| Print | @media print CSS verified |
| Mobile viewport | Responsive layout at 768px breakpoint |

---

## 11. CI/CD Integration

### 11.1 GitHub Actions Workflow

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  backend-unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: make init
      - run: make test-unit

  backend-integration:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: demand_test
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: make init
      - run: make test-integration

  frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: make ui-init
      - run: make ui-test

  type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: cd mvp/demand/frontend && npx tsc --noEmit
```

### 11.2 Coverage Thresholds

| Scope | Minimum Coverage | Target Coverage |
|---|---|---|
| `common/` modules | 90% | 95% |
| `api/main.py` | 70% | 85% |
| Frontend utilities (`lib/`) | 90% | 95% |
| Frontend hooks | 80% | 90% |
| Frontend components | 60% | 75% |
| Frontend tabs | 50% | 70% |

### 11.3 Pre-commit Checks

```yaml
# Minimum gates before merge:
- All unit tests pass
- TypeScript compiles with no errors (tsc --noEmit)
- Python type hints valid (mypy --strict on common/)
- No new test coverage regressions
```

---

## 12. Testing Requirements for New Development

**Every new feature or significant change MUST include tests. This is a mandatory part of the development workflow.**

### 12.1 When Tests Are Required

| Change Type | Test Requirement |
|---|---|
| New API endpoint | Unit test for handler logic + integration test with test client |
| New frontend component | Component test (render, props, interactions, loading/error states) |
| New custom hook | Hook test (initial state, state transitions, cleanup) |
| New utility function | Unit test (happy path, edge cases, error cases) |
| New Python script | Unit test for core logic; integration test if it touches DB |
| New common module function | Unit test with ≥ 90% branch coverage |
| Bug fix | Regression test that reproduces the bug before fix |
| New backtest model | Test for model-specific training function + prediction shape validation |
| Schema change (new table/column) | Data integrity test for new constraints |
| New Make target | Verify it runs without error in CI |

### 12.2 Test Checklist for Pull Requests

Every PR should include (enforced via review):

```markdown
## Test Checklist
- [ ] Unit tests added for new/changed functions
- [ ] Component tests added for new/changed UI components
- [ ] Integration tests added for new API endpoints
- [ ] Edge cases covered (null, empty, boundary values)
- [ ] Error states tested (API failures, invalid input)
- [ ] Loading states tested (skeleton/spinner shown)
- [ ] No flaky tests (deterministic, no timing dependencies)
- [ ] Existing tests still pass (`make test-all`)
- [ ] TypeScript compiles (`tsc --noEmit`)
```

### 12.3 Test Writing Guidelines

**Naming convention:**
```
# Python
test_{function_name}_{scenario}_{expected_outcome}
# Example: test_compute_accuracy_metrics_empty_series_returns_none

# TypeScript
it("should {expected behavior} when {condition}")
// Example: it("should return dash when value is null")
```

**Test structure (Arrange-Act-Assert):**
```python
# Python
def test_compute_accuracy_metrics_perfect_forecast():
    # Arrange
    forecast = pd.Series([100, 200, 300])
    actual = pd.Series([100, 200, 300])

    # Act
    result = compute_accuracy_metrics(forecast, actual)

    # Assert
    assert result["wape"] == 0
    assert result["accuracy_pct"] == 100
    assert result["bias"] == 0
```

```typescript
// TypeScript
it("should render all 7 domains in selector", async () => {
  // Arrange
  render(<ExplorerTab />);

  // Act
  const selector = screen.getByRole("combobox");
  await userEvent.click(selector);

  // Assert
  expect(screen.getAllByRole("option")).toHaveLength(7);
});
```

**Mocking guidelines:**
- Mock at the boundary (HTTP calls, DB connections, external APIs)
- Never mock the function under test
- Use `vi.fn()` for simple mocks, MSW for API mocking
- Reset mocks between tests (`afterEach(() => vi.restoreAllMocks())`)

### 12.4 What NOT to Test

- Third-party library internals (Recharts rendering, shadcn/ui styling)
- CSS class names or DOM structure (test behavior, not implementation)
- Exact error message strings (test error type/category)
- Private functions (test through public API)
- Generated code (Pydantic models, TypeScript types)

---

## 13. Test Prioritization (Implementation Order)

### Phase 1 — Foundation (Week 1-2)

**Goal:** Establish testing infrastructure and cover critical calculation logic.

| Priority | Area | Tests | Estimated Count |
|---|---|---|---|
| P0 | `common/metrics.py` | All accuracy calculations | 10 |
| P0 | `common/feature_engineering.py` | Feature matrix correctness, no future leakage | 12 |
| P0 | `common/backtest_framework.py` | Timeframes, lag assignment, post-processing | 15 |
| P0 | API test infrastructure | conftest.py, test client, fixtures | — |
| P0 | Frontend test infrastructure | MSW setup, query wrapper, render helpers | — |
| P1 | `common/domain_specs.py` | Schema validation | 5 |
| P1 | `common/constants.py` | Bounds and invariants | 4 |

### Phase 2 — API Coverage (Week 3-4)

| Priority | Area | Tests | Estimated Count |
|---|---|---|---|
| P0 | Domain endpoints | CRUD, pagination, search, filters | 20 |
| P0 | Accuracy endpoints | KPIs, slicing, lag curve | 15 |
| P0 | DFU Analysis endpoint | 3 modes, multi-model, edge cases | 10 |
| P1 | Competition endpoints | Config, run, summary | 10 |
| P1 | Cluster endpoints | List, profiles, visualization | 6 |
| P2 | Chat endpoint | NL→SQL with mocked OpenAI | 6 |
| P2 | Market intelligence | Mocked Google + OpenAI | 5 |

### Phase 3 — Frontend Components (Week 5-6)

| Priority | Area | Tests | Estimated Count |
|---|---|---|---|
| P0 | Custom hooks | useTheme, useUrlState, useKeyboardShortcuts | 20 |
| P0 | DataTable component | Rendering, sorting, filtering, virtualization | 15 |
| P0 | API query layer | Query keys, fetch functions, error handling | 10 |
| P1 | ExplorerTab | Domain selection, search, pagination, export | 15 |
| P1 | AccuracyTab | KPI cards, model selector, charts | 12 |
| P1 | DfuAnalysisTab | Mode selection, chart rendering, KPIs | 12 |
| P2 | ClustersTab | Table, profiles | 5 |
| P2 | MarketIntelTab | Form, results, narrative | 5 |
| P2 | ChatPanel | Input, messages, SQL display | 6 |

### Phase 4 — Scripts & Integration (Week 7-8)

| Priority | Area | Tests | Estimated Count |
|---|---|---|---|
| P1 | Normalize script | Null handling, type casting, idempotency | 10 |
| P1 | Load script | Row counts, upsert, timestamps | 8 |
| P1 | Clustering pipeline | Feature gen, training, labeling | 10 |
| P1 | Champion selection | Winner logic, ceiling oracle | 8 |
| P2 | Integration: data pipeline | End-to-end CSV → query | 5 |
| P2 | Integration: backtest pipeline | Mini backtest → load → query | 5 |

### Phase 5 — Security, Performance, Accessibility (Ongoing)

| Priority | Area | Tests | Estimated Count |
|---|---|---|---|
| P1 | SQL injection prevention | Parameterized queries verified | 5 |
| P1 | Input validation | Bounds, types, injection | 8 |
| P2 | API latency benchmarks | Response time assertions | 5 |
| P2 | Accessibility | Keyboard nav, aria labels, contrast | 8 |

**Total estimated test count: ~350 tests**

---

## 14. Dependencies to Install

### Python (add to `pyproject.toml` dev dependencies)

```toml
[dependency-groups]
dev = [
    "pytest>=8.3",
    "pytest-asyncio>=0.24",
    "pytest-cov>=6.0",
    "pytest-mock>=3.14",
    "httpx>=0.28",           # ASGI test client for FastAPI
]
```

### Frontend (add to `package.json` devDependencies)

```json
{
  "devDependencies": {
    "msw": "^2.6.0",
    "@testing-library/user-event": "^14.5.0"
  }
}
```

(Already installed: `vitest`, `@testing-library/react`, `@testing-library/jest-dom`, `jsdom`)

---

## 15. Success Criteria

| Metric | Target |
|---|---|
| All unit tests pass | 100% |
| `common/` module coverage | ≥ 90% |
| API endpoint coverage | ≥ 70% |
| Frontend utility coverage | ≥ 90% |
| Frontend component coverage | ≥ 60% |
| CI pipeline runs in | < 5 minutes |
| Zero flaky tests | 0 failures on re-run |
| Every new PR has tests | Enforced via checklist |

---

## Implementation Status: Fully Implemented

The testing infrastructure is fully operational. Actual test count: **485+ tests** (backend + frontend), significantly exceeding the original estimate of ~350.

### Actual Test Directory Structure
- `tests/unit/` and `tests/api/` (no `tests/scripts/` or `tests/integration/` directories)

### Additional Backend Test Files (not in original spec)
- `tests/unit/test_db.py`, `test_scenario_runner.py`, `test_seasonality.py`, `test_inventory_domain.py`, `test_champion_selection.py`
- `tests/api/test_dashboard.py`, `test_distinct.py`, `test_inventory.py`, `test_inventory_backtest.py`, `test_clustering_scenario.py`, `test_jobs.py`, `test_seasonality.py`

### Additional Frontend Test Files (not in original spec)
- Components: `MotifSettingsPanel`, `ElementTab`, `LoadingElement`, `ThemeSelector`, `WidgetGrid`, `AlertPanel`, `TopMovers`, `HeatmapGrid`, `GlobalFilterBar`, `AppSidebar`
- Hooks: `useMotifTheme`, `useSidebar`, `useGlobalFilters`
- Context: `ScenarioNotificationContext`, `JobNotificationContext`
- Tabs: `DashboardTab`, `InventoryTab`, `InvBacktestTab`, `WhatIfScenarios`, `JobsTab`
- Constants: `motifRegistry`

### Fixture Corrections
- Backend: `sample_dfu_attrs` (not `sample_dfu_df`), `sample_item_attrs` (not in original spec)
- Frontend: `test-utils.tsx` provides `TestQueryWrapper` for wrapping with `QueryClientProvider`

### Domain Count
- 8 domains (not 7): item, location, customer, time, dfu, sales, forecast, inventory


---

## Examples

### Example: Backend API test with ASGI transport

```python
# tests/api/test_competition.py
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.mark.asyncio
async def test_competition_results(mock_pool):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/competition/results?lag=2&model=lgbm_global")
    assert resp.status_code == 200
    data = resp.json()
    assert "rows" in data
    assert all("accuracy_pct" in r for r in data["rows"])
```

### Example: Frontend component test (Vitest + RTL)

```typescript
// src/components/__tests__/KpiCard.test.tsx
import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { KpiCard } from '@/components/KpiCard'

describe('KpiCard', () => {
  it('renders accuracy metric with correct value', () => {
    render(<KpiCard label="Accuracy" value={92.5} unit="%" trend="up" />)
    expect(screen.getByText('92.5%')).toBeInTheDocument()
    expect(screen.getByText('Accuracy')).toBeInTheDocument()
  })
})
```

### Example: Run all tests

```bash
make test-all
# Backend:  pytest tests/ -x  (~0.7s, no infra needed — DB mocked)
# Frontend: vitest run         (~1.5s, 218 tests)
# Total: ~2.2 seconds

make test-cov   # backend coverage report
# Coverage: 87% (common/), 84% (api/routers/)
```
