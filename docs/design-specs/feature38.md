# Feature 38: Clustering What-If Scenario Enhancements — Background Execution, Runtime Estimation, Dashboard Alerts, Enhanced Charts

## Executive Summary

Feature 38 enhances the Clustering What-If Scenarios panel (Feature 29) with four improvements:
1. **Runtime Estimation** — display approximate execution time before running a scenario
2. **Background Execution** — non-blocking async POST with polling, so users can navigate away
3. **Dashboard Alerts** — notification on the Dashboard tab when a scenario completes
4. **Enhanced Charts** — richer elbow/silhouette/feature importance visualizations with quality indicators

## Key Features

- **Estimate Endpoint:** `GET /clustering/scenario/estimate` — returns estimated runtime based on DFU count, K range, and gap statistic flag
- **Async POST:** `POST /clustering/scenario` now returns HTTP 202 immediately with `scenario_id` and runs in background
- **Status Polling:** `GET /clustering/scenario/{id}/status` — returns `running` (with elapsed time), `completed` (with full result), or `failed`
- **ScenarioNotificationContext** — React context for cross-tab scenario state tracking
- **Dashboard Alert Injection** — scenario completion alert prepended to Dashboard AlertPanel with dismiss support
- **Enhanced Elbow Chart** — optimal K reference line with marker
- **Enhanced Silhouette Chart** — bar chart with quality zone thresholds (Strong/Reasonable/Weak/No structure), color-coded bars
- **Feature Importance Chart** — horizontal bar chart showing top 10 features by variance ratio
- **Cluster Size Pie Chart** — replaces basic bar chart with labeled pie chart
- **Gap Statistic Chart** — conditional line chart when gap stats are available

---

## API Endpoints

### `GET /clustering/scenario/estimate`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `k_min` | int | 3 | Min K value |
| `k_max` | int | 12 | Max K value |
| `skip_gap` | bool | true | Skip gap statistic calculation |

**Response:**
```json
{ "estimated_seconds": 45, "dfu_count": 1200, "k_range": 10, "skip_gap": true }
```

### `POST /clustering/scenario` (Updated)

Now returns **HTTP 202** immediately instead of blocking:
```json
{ "scenario_id": "sc_20260226_abc1", "status": "running" }
```

### `GET /clustering/scenario/{id}/status`

**While running:**
```json
{ "scenario_id": "sc_123", "status": "running", "elapsed_seconds": 12 }
```

**When complete:**
```json
{ "scenario_id": "sc_123", "status": "completed", "runtime_seconds": 45, "result": { ... } }
```

---

## UI Changes

### ClustersTab
- **Estimate badge** next to Run button: "Est. ~45s (1.2K DFUs)"
- **Running indicator** with elapsed time and spinner animation
- **Background polling** every 3s via TanStack Query `refetchInterval`

### Dashboard AlertPanel
- Scenario completion alert with FlaskConical icon and dismiss button
- Alert text: "Scenario {label} Complete — finished in {time}s. View results in Clusters tab."

### Enhanced ScenarioCharts
- Elbow: ReferenceLine at optimal K
- Silhouette: Bar chart with quality zone reference lines, Cell color-coding
- Feature Importance: Horizontal bar chart (top 10)
- Cluster Size: PieChart with labels
- Gap Statistic: Conditional LineChart with optimal K marker

---

## Testing

### Backend Tests (14 total in test_clustering_scenario.py)
- 7 existing tests (updated: POST now returns 202)
- 7 new tests: estimate (3), status (3), conflict (1)

### Frontend Tests
- ScenarioNotificationContext: 4 tests (defaults, start, complete, dismiss)
- ClustersTab: 3 tests (smoke, cluster summary, what-if section)
- DashboardTab: 4 tests (updated with ScenarioNotificationProvider)

---

## Files

| File | Action |
|------|--------|
| `api/routers/clusters.py` | Edited — estimate endpoint, async POST, status endpoint |
| `scripts/run_clustering_scenario.py` | Unchanged — result saving already in place |
| `frontend/src/api/queries.ts` | Edited — estimate + status fetch functions, query keys |
| `frontend/src/tabs/ClustersTab.tsx` | Edited — estimation UI, polling, enhanced ScenarioCharts |
| `frontend/src/context/ScenarioNotificationContext.tsx` | **Created** — cross-tab notification context |
| `frontend/src/App.tsx` | Edited — ScenarioNotificationProvider wrapper |
| `frontend/src/types/theme.ts` | Edited — scenario_complete AlertType |
| `frontend/src/components/AlertPanel.tsx` | Edited — FlaskConical icon, dismiss button, click handler |
| `frontend/src/tabs/DashboardTab.tsx` | Edited — scenario alert injection |
| `tests/api/test_clustering_scenario.py` | Edited — 14 tests (7 new) |
| `frontend/src/context/__tests__/ScenarioNotificationContext.test.tsx` | **Created** — 4 tests |
| `frontend/src/tabs/__tests__/ClustersTab.test.tsx` | Edited — updated with provider |
| `frontend/src/tabs/__tests__/DashboardTab.test.tsx` | Edited — updated with provider |

---

## Implementation Corrections

### Estimate Response
2 additional fields not in spec:
- `training_sample` (int) — number of DFUs used for training (capped at 20,000)
- `sampled` (boolean) — whether DFU count exceeds sampling threshold
- Undocumented `scope` query parameter on estimate endpoint

### POST Scenario Response
- Additional field: `job_id` (for tracking in Jobs tab)

### JobManager Integration (Feature 39)
- POST handler delegates to `JobManager.submit_job("cluster_scenario", ...)` instead of running inline
- Maintains legacy state tracking for backward compatibility with status polling
- 409 conflict raised via `RuntimeError` from `manager.submit_job()` (not direct `_scenario_running` check)

### Runtime Estimation Formula
- `feature_gen_per_dfu = 0.001s`, `kmeans_per_dfu_per_k = 0.002s`
- `gap_multiplier = 2.5x`, `overhead_seconds = 10.0`
- `max_training_dfus = 20,000` sampling cap

### ScenarioNotificationContext
- `CompletedScenario` interface: `id`, `label`, `runtimeSeconds`, `result`
- `useScenarioNotification()` hook with error if used outside provider
- `failScenario()` method for error handling

### Additional Endpoint
- `GET /clustering/scenario/{scenario_id}` — retrieve scenario result directly (separate from `/status`)

### Pydantic Models (not in spec)
- `FeatureParams`: time_window_months, min_months_history
- `ModelParams`: k_range, min_cluster_size_pct, use_pca, pca_components, skip_gap, all_features
- `LabelParams`: volume_high, volume_low, cv_steady, cv_volatile, seasonality_threshold, zero_demand_threshold
- `ClusteringScenarioRequest`: feature_params, model_params, label_params, relabel_only, previous_scenario_id


---

## Examples

### Example: Full enhanced scenario workflow

```bash
# 1. Get runtime estimate
curl -s "http://localhost:8000/clustering/scenario/estimate?k_max=8&gap=true" | jq .
# {"estimated_seconds": 72, "dfu_count": 18432, "k_range": 5, "skip_gap": false}

# 2. Submit (202 Accepted — non-blocking)
SCENARIO_ID=$(curl -s -X POST http://localhost:8000/clustering/scenario \
  -H "Content-Type: application/json" \
  -d '{"model_params": {"k_range": [3, 8], "use_pca": false}, "feature_params": {}, "label_params": {}}' \
  | jq -r '.id')

# 3. Poll until complete (check every 3s)
until curl -s "http://localhost:8000/clustering/scenario/$SCENARIO_ID/status" | jq -e '.status == "completed"' > /dev/null; do
  sleep 3
done

# 4. Promote winning scenario to production
curl -s -X POST "http://localhost:8000/clustering/scenario/$SCENARIO_ID/promote"
# {"promoted": true, "dfus_updated": 18432, "best_k": 6, "silhouette": 0.71}
```

### Example: Enhanced chart descriptions

- **Elbow chart**: WCSS vs K with `ReferenceLine` at optimal K (red dashed vertical line)
- **Silhouette chart**: Bar chart with quality zones: Strong (≥0.7), Reasonable (0.5-0.7), Weak (0.25-0.5)
- **Feature importance**: Horizontal bars showing top 10 features driving cluster separation
- **Cluster size pie**: Pie chart with percentage labels, n_dfus per cluster
- **Gap statistic**: Line chart comparing gap statistic vs reference — shown only when `skip_gap=False`

### Example: Scenario queueing when group is busy

```bash
# Submit 2 scenarios while one is running → second gets queued
curl -s -X POST http://localhost:8000/clustering/scenario -d '{"model_params": {}}' | jq .status
# "running"
curl -s -X POST http://localhost:8000/clustering/scenario -d '{"model_params": {"k_range": [4,6]}}' | jq .status
# "queued"  ← auto-dispatched when running job completes
```
