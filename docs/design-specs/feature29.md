# Feature 29 — What-If / Scenario UI for Clustering

## Overview

Add an interactive **What-If Scenarios** panel to the Clusters tab that lets users adjust clustering parameters, run trial configurations against the backend pipeline, compare results side-by-side, and optionally promote a chosen scenario to production. This transforms clustering from a static, CLI-only workflow into a self-service analytics experience.

## Problem

Today the clustering pipeline is entirely CLI-driven:

1. **No parameter visibility** — users see the final cluster table and static PNGs but have no way to understand how K, time window, feature set, or labeling thresholds produced those results
2. **No experimentation** — changing a single parameter (e.g., K range, PCA toggle, CV threshold) requires editing `clustering_config.yaml`, re-running `make cluster-all`, and waiting for the full pipeline
3. **No comparison** — there is no way to compare two configurations side-by-side (e.g., "K=5 with PCA" vs "K=8 without PCA") to evaluate trade-offs
4. **No interactive charts** — the K-selection data (`k_values`, `inertias`, `silhouette_scores`, `gap_stats`) is already returned by the `/domains/dfu/clusters/profiles` API but is displayed only as static PNGs; users cannot hover, zoom, or overlay multiple runs
5. **No labeling tuning** — the volume/CV/seasonality thresholds in the labeling step directly control business labels but can only be changed in YAML

## Goals

- Expose all meaningful clustering parameters in an interactive UI
- Let users run multiple trial scenarios without affecting production clusters
- Provide rich interactive charts (Recharts) replacing the static PNGs
- Enable side-by-side scenario comparison with visual and quantitative diffs
- Allow promotion of a selected scenario to production (write to `dim_dfu.ml_cluster`)

## Non-Goals (Out of Scope)

- Changing the clustering algorithm itself (remains KMeans)
- Real-time streaming of pipeline progress (polling is sufficient)
- Undo/rollback of promoted scenarios (manual re-run via CLI serves this)
- Persisting scenario history across browser sessions (in-memory only; MLflow already tracks runs)

---

## Architecture

### Data Flow

```
User adjusts sliders/toggles in What-If panel
        |
        v
POST /clustering/scenario  (new endpoint)
        |
        v
Backend runs pipeline stages in a temp directory:
  1. generate_clustering_features.py --time-window X --min-months Y --output /tmp/scenario_<id>/
  2. train_clustering_model.py --k-range MIN MAX --use-pca --min-cluster-size-pct Z ...
  3. label_clusters.py (with custom thresholds)
        |
        v
Returns scenario result JSON:
  - cluster_assignments (count per cluster)
  - cluster_centroids (feature means)
  - cluster_profiles (label + centroid features)
  - k_selection_results (for interactive charts)
  - metadata (silhouette, inertia, optimal_k, runtime)
        |
        v
Frontend renders interactive charts + summary table
        |
        v (optional)
POST /clustering/scenario/<id>/promote
        |
        v
Runs label + update_cluster_assignments against dim_dfu
```

### API Endpoints (New)

#### `POST /clustering/scenario`

Runs a trial clustering pipeline with user-specified parameters. Results are stored server-side in a temp directory keyed by scenario ID. Does **not** modify `dim_dfu`.

**Request body:**

```json
{
  "feature_params": {
    "time_window_months": 24,
    "min_months_history": 6
  },
  "model_params": {
    "k_range": [3, 10],
    "min_cluster_size_pct": 2.0,
    "use_pca": false,
    "pca_components": null,
    "skip_gap": true,
    "all_features": false
  },
  "label_params": {
    "volume_high": 0.75,
    "volume_low": 0.25,
    "cv_steady": 0.3,
    "cv_volatile": 0.8,
    "seasonality_threshold": 0.5,
    "zero_demand_threshold": 0.2
  }
}
```

All fields are optional — omitted fields use the defaults from `clustering_config.yaml`.

**Response:**

```json
{
  "scenario_id": "sc_20260222_143022_a1b2",
  "status": "completed",
  "runtime_seconds": 14.3,
  "params": { "...merged params with defaults..." },
  "result": {
    "optimal_k": 6,
    "silhouette_score": 0.412,
    "inertia": 8341.2,
    "n_clusters": 6,
    "total_dfus": 4821,
    "cluster_sizes": { "0": 1203, "1": 892 },
    "k_selection_results": {
      "k_values": [3, 4, 5, 6, 7, 8, 9, 10],
      "inertias": [18432, 14221, 11003, 8341, 7102, 6244, 5811, 5503],
      "silhouette_scores": [0.31, 0.35, 0.39, 0.41, 0.38, 0.36, 0.33, 0.31],
      "gap_stats": null
    },
    "profiles": [
      {
        "cluster_id": 0,
        "label": "high_volume_steady",
        "count": 1203,
        "pct_of_total": 24.95,
        "mean_demand": 387.2,
        "cv_demand": 0.18,
        "seasonality_strength": 0.12,
        "trend_slope": 0.003,
        "growth_rate": 1.2,
        "zero_demand_pct": 0.01
      }
    ],
    "feature_importance": [
      { "feature": "mean_demand", "variance_ratio": 0.34 },
      { "feature": "cv_demand", "variance_ratio": 0.22 }
    ]
  }
}
```

#### `GET /clustering/scenario/<scenario_id>`

Retrieve a previously run scenario result. Returns 404 if the temp directory has been cleaned up.

#### `POST /clustering/scenario/<scenario_id>/promote`

Promotes a scenario to production:
1. Copies the scenario's `cluster_labels.csv` to `data/clustering/`
2. Runs `update_cluster_assignments.py` to write labels to `dim_dfu.ml_cluster`
3. Refreshes the cluster summary cache

**Response:**

```json
{
  "status": "promoted",
  "scenario_id": "sc_20260222_143022_a1b2",
  "dfus_updated": 4821,
  "cluster_distribution": { "high_volume_steady": 1203 }
}
```

#### `GET /clustering/defaults`

Returns the current default parameter values from `clustering_config.yaml` so the UI can populate sliders with baseline values.

**Response:**

```json
{
  "feature_params": {
    "time_window_months": 24,
    "min_months_history": 1
  },
  "model_params": {
    "k_range": [3, 12],
    "min_cluster_size_pct": 2.0,
    "use_pca": false,
    "pca_components": null,
    "skip_gap": false,
    "all_features": false
  },
  "label_params": {
    "volume_high": 0.75,
    "volume_low": 0.25,
    "cv_steady": 0.3,
    "cv_volatile": 0.8,
    "seasonality_threshold": 0.5,
    "zero_demand_threshold": 0.2
  }
}
```

---

## UI Design

### Panel Location

The What-If panel lives within the existing **Clusters tab** (`Cl` element tile), rendered below the current cluster summary table and visualization section. It is collapsed by default behind a disclosure button: **"What-If Scenarios"**.

### Layout

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Cl  Clusters                                                          │
│─────────────────────────────────────────────────────────────────────────│
│  Source: [ML Pipeline ▾]    Cluster: [All ▾]                           │
│  6 clusters, 4821 DFUs assigned                                        │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Cluster │ DFUs │   %  │ Avg Demand │  CV  │                     │   │
│  │─────────┼──────┼──────┼────────────┼──────│                     │   │
│  │ high_v… │ 1203 │ 24.9 │    387.2   │ 0.18 │  ← existing table  │   │
│  │ ...     │      │      │            │      │                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  K = 6   Silhouette = 0.4120   Inertia = 8341  ← existing badges      │
│                                                                         │
│  ▼ What-If Scenarios ─────────────────────────────────────────────────  │
│                                                                         │
│  ┌──────────────── Parameter Controls ─────────────────────────────┐   │
│  │                                                                  │   │
│  │  DATA SCOPE                                                      │   │
│  │  Time Window (months)   [══════●══════] 24                       │   │
│  │  Min History (months)   [●═════════════]  1                      │   │
│  │                                                                  │   │
│  │  MODEL                                                           │   │
│  │  K Range                [══●═══════●══] 3 – 12                   │   │
│  │  Min Cluster Size (%)   [══●══════════] 2.0                      │   │
│  │  Use PCA                [ ] Off                                  │   │
│  │  PCA Components         [══════════●══] auto  (disabled if off)  │   │
│  │  Skip Gap Statistic     [✓] On                                   │   │
│  │  Feature Set            (●) Core 8   ( ) All Features            │   │
│  │                                                                  │   │
│  │  LABELING THRESHOLDS                                             │   │
│  │  Volume High (pctl)     [═══════════●═] 0.75                     │   │
│  │  Volume Low (pctl)      [══●══════════] 0.25                     │   │
│  │  CV Steady (<)          [═══●═════════] 0.30                     │   │
│  │  CV Volatile (>)        [═══════════●═] 0.80                     │   │
│  │  Seasonality Threshold  [═══════●═════] 0.50                     │   │
│  │  Zero Demand Threshold  [═══●═════════] 0.20                     │   │
│  │                                                                  │   │
│  │  [  Reset to Defaults  ]          [ ▶ Run Scenario ]             │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────── Scenario Results ───────────────────────────────┐   │
│  │                                                                  │   │
│  │  Scenario A (14.3s)              vs    Scenario B (11.8s)        │   │
│  │  K=6  Sil=0.412  Inertia=8341         K=5  Sil=0.391  Ine=11003│   │
│  │                                                                  │   │
│  │  ┌─ K-Selection Chart (interactive Recharts) ─────────────────┐ │   │
│  │  │                                                             │ │   │
│  │  │   Elbow (WCSS)           Silhouette Score                  │ │   │
│  │  │   ┌──────────┐           ┌──────────┐                      │ │   │
│  │  │   │  ╲       │           │    ╱╲    │     ── Scenario A    │ │   │
│  │  │   │   ╲__    │           │   ╱  ╲   │     -- Scenario B    │ │   │
│  │  │   │      ╲___│           │__╱    ╲__│                      │ │   │
│  │  │   └──────────┘           └──────────┘                      │ │   │
│  │  │   3  4  5  6  7  8       3  4  5  6  7  8                  │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  ┌─ Cluster Profile Radar Chart ──────────────────────────────┐ │   │
│  │  │          mean_demand                                        │ │   │
│  │  │             ╱╲                                              │ │   │
│  │  │  zero_pct ╱    ╲ cv_demand                                 │ │   │
│  │  │          ╱  C0  ╲                                           │ │   │
│  │  │ growth ─┤   C1   ├─ seasonality     (one polygon per       │ │   │
│  │  │          ╲  C2  ╱    cluster, hover to highlight)           │ │   │
│  │  │           ╲    ╱                                            │ │   │
│  │  │            ╲╱                                               │ │   │
│  │  │         trend_slope                                         │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  ┌─ Cluster Size Distribution (bar chart) ────────────────────┐ │   │
│  │  │  ████████████████  high_volume_steady (1203, 25%)          │ │   │
│  │  │  ████████████      medium_volume_seasonal (892, 19%)       │ │   │
│  │  │  ██████████        low_volume_intermittent (714, 15%)      │ │   │
│  │  │  █████████         medium_volume_steady (683, 14%)         │ │   │
│  │  │  ████████          high_volume_growing (641, 13%)          │ │   │
│  │  │  ███████           low_volume_declining (688, 14%)         │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  ┌─ Comparison Table ─────────────────────────────────────────┐ │   │
│  │  │ Metric           │ Scenario A │ Scenario B │  Delta        │ │   │
│  │  │──────────────────┼────────────┼────────────┼───────────────│ │   │
│  │  │ Optimal K        │     6      │     5      │  -1           │ │   │
│  │  │ Silhouette       │   0.412    │   0.391    │  -0.021 ▼     │ │   │
│  │  │ Inertia          │   8,341    │  11,003    │  +2,662 ▲     │ │   │
│  │  │ Total DFUs       │   4,821    │   4,821    │  —            │ │   │
│  │  │ Largest Cluster  │  25.0%     │  31.2%     │  +6.2pp       │ │   │
│  │  │ Smallest Cluster │  13.1%     │  14.8%     │  +1.7pp       │ │   │
│  │  │ Runtime          │  14.3s     │  11.8s     │  -2.5s        │ │   │
│  │  └─────────────────────────────────────────────────────────────┘ │   │
│  │                                                                  │   │
│  │  [ ★ Promote Scenario A ]  [ ★ Promote Scenario B ]             │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Parameter Controls — Detail

#### Data Scope Section

| Control | Type | Range | Default | Step | Description |
|---------|------|-------|---------|------|-------------|
| Time Window (months) | Slider | 6 – 60, plus "All" toggle | 24 | 6 | Sales lookback window for feature engineering |
| Min History (months) | Slider | 1 – 24 | 1 | 1 | Minimum months of sales data required to include a DFU |

#### Model Section

| Control | Type | Range | Default | Step | Description |
|---------|------|-------|---------|------|-------------|
| K Range Min | Dual-thumb slider | 2 – 20 | 3 | 1 | Lower bound of K search |
| K Range Max | Dual-thumb slider | 2 – 20 | 12 | 1 | Upper bound of K search (must be > min) |
| Min Cluster Size (%) | Slider | 0.5 – 10.0 | 2.0 | 0.5 | Clusters smaller than this % get merged |
| Use PCA | Toggle switch | on/off | off | — | Enable PCA dimensionality reduction before KMeans |
| PCA Components | Slider | 2 – 8, plus "auto" | auto | 1 | Number of PCA components (disabled when PCA is off) |
| Skip Gap Statistic | Toggle switch | on/off | off | — | Skip gap stat calculation (faster but less info) |
| Feature Set | Radio group | Core 8 / All | Core 8 | — | Core 8 features vs all numeric features |

#### Labeling Thresholds Section

| Control | Type | Range | Default | Step | Description |
|---------|------|-------|---------|------|-------------|
| Volume High Percentile | Slider | 0.50 – 0.95 | 0.75 | 0.05 | Centroid mean_demand above this percentile = "high_volume" |
| Volume Low Percentile | Slider | 0.05 – 0.50 | 0.25 | 0.05 | Centroid mean_demand below this percentile = "low_volume" |
| CV Steady Threshold | Slider | 0.1 – 0.5 | 0.30 | 0.05 | CV below this = "steady" pattern |
| CV Volatile Threshold | Slider | 0.5 – 1.5 | 0.80 | 0.05 | CV above this = "volatile" pattern |
| Seasonality Threshold | Slider | 0.1 – 1.0 | 0.50 | 0.05 | seasonality_strength above this = "seasonal" |
| Zero Demand Threshold | Slider | 0.05 – 0.50 | 0.20 | 0.05 | zero_demand_pct above this = "intermittent" |

#### Validation Rules

- `k_range[0] < k_range[1]` (min < max)
- `volume_low < volume_high`
- `cv_steady < cv_volatile`
- `pca_components` disabled when `use_pca` is false
- Display inline validation errors below controls when violated

### Scenario Slots

The UI maintains up to **2 scenario slots** (A and B) for comparison:

- **First run** populates Slot A
- **Second run** populates Slot B (Slot A preserved for comparison)
- **Third run** replaces the oldest slot (user chooses which to replace, or auto-replaces A)
- Each slot displays a compact summary badge: `Scenario A: K=6, Sil=0.41, 14.3s`
- Active slot is highlighted; both are always visible in the comparison section

### Interactive Charts

All charts use **Recharts** (consistent with the rest of the app) and support:
- Hover tooltips with exact values
- Theme-aware colors via `CHART_COLORS[theme]`
- Legend toggle to show/hide individual series
- Responsive container for window resizing

#### 1. K-Selection Chart (replaces static PNG)

A **2-panel** line chart (or 3-panel if gap stats are available):

**Left panel — Elbow (WCSS/Inertia):**
- X-axis: K values
- Y-axis: Inertia (WCSS)
- Line per scenario (solid for A, dashed for B)
- Vertical dashed line at each scenario's optimal K
- Tooltip: `K=6, Inertia=8,341`

**Right panel — Silhouette Score:**
- X-axis: K values
- Y-axis: Silhouette score (0–1)
- Line per scenario
- Highlighted peak point (optimal K)
- Tooltip: `K=6, Silhouette=0.412`

**Optional third panel — Gap Statistic** (shown only when `skip_gap=false`):
- X-axis: K values
- Y-axis: Gap statistic
- Same overlay pattern

#### 2. Cluster Profile Radar Chart

A **Recharts RadarChart** with one polygon per cluster:

- Axes: `mean_demand`, `cv_demand`, `seasonality_strength`, `trend_slope`, `growth_rate`, `zero_demand_pct`
- Values normalized to 0–1 range (min-max across all clusters in the scenario)
- One colored polygon per cluster (colors from `MODEL_COLORS` or `tab10`-style palette)
- Hover highlights individual cluster polygon and shows centroid values
- Legend: cluster labels with color swatches
- If comparing two scenarios: a tab/toggle to switch between Scenario A and B radar views

#### 3. Cluster Size Distribution (Horizontal Bar Chart)

A **Recharts BarChart** (horizontal layout):

- Y-axis: Cluster labels (e.g., `high_volume_steady`)
- X-axis: Count (or percentage of total)
- Bars colored by cluster
- If comparing: grouped bars (A and B side-by-side per cluster label)
- Tooltip: `high_volume_steady: 1,203 DFUs (25.0%)`

#### 4. Comparison Table

A standard HTML table (matching existing project table style):

| Metric | Scenario A | Scenario B | Delta |
|--------|-----------|-----------|-------|
| Optimal K | 6 | 5 | -1 |
| Silhouette Score | 0.4120 | 0.3910 | -0.0210 (red down arrow) |
| Inertia | 8,341 | 11,003 | +2,662 (red up arrow — lower is better) |
| Total DFUs | 4,821 | 4,821 | — |
| Largest Cluster % | 25.0% | 31.2% | +6.2pp |
| Smallest Cluster % | 13.1% | 14.8% | +1.7pp |
| Balance (std of sizes) | 184 | 211 | +27 |
| Runtime | 14.3s | 11.8s | -2.5s |

Delta coloring: green = better, red = worse, gray = neutral. Direction depends on the metric (lower inertia is better, higher silhouette is better).

### Loading State

When a scenario is running, the **Run Scenario** button shows a spinner and the results area displays a `LoadingElement` component (existing chemistry-themed periodic table tile with pulse-glow animation) with the message: **"Running clustering scenario..."**

Estimated runtime varies by parameters:
- Core features, skip gap: ~10–20s
- Core features, with gap: ~30–60s
- All features, with gap: ~60–120s

The UI displays an estimated time range based on the selected options.

### Promote Flow

When the user clicks **"Promote Scenario X"**:

1. A confirmation dialog appears:
   ```
   ┌─────────────────────────────────────────────┐
   │  Promote Scenario A to Production?           │
   │                                               │
   │  This will update ml_cluster for 4,821 DFUs  │
   │  in dim_dfu with the following configuration: │
   │                                               │
   │  K = 6 | Silhouette = 0.412                   │
   │  Time Window = 24 months                      │
   │  Features = Core 8 | PCA = Off                │
   │                                               │
   │         [ Cancel ]    [ Promote ]              │
   └─────────────────────────────────────────────┘
   ```

2. On confirm, `POST /clustering/scenario/<id>/promote` is called
3. Success toast: "Cluster assignments updated for 4,821 DFUs"
4. The main cluster summary table (above the What-If panel) auto-refreshes to reflect the new assignments
5. MLflow run is logged with scenario parameters + "promoted" tag

---

## State Management

### Frontend State (within Clusters tab)

```typescript
// What-If panel state
interface ScenarioParams {
  feature_params: {
    time_window_months: number | "all";
    min_months_history: number;
  };
  model_params: {
    k_range: [number, number];
    min_cluster_size_pct: number;
    use_pca: boolean;
    pca_components: number | null;
    skip_gap: boolean;
    all_features: boolean;
  };
  label_params: {
    volume_high: number;
    volume_low: number;
    cv_steady: number;
    cv_volatile: number;
    seasonality_threshold: number;
    zero_demand_threshold: number;
  };
}

interface ScenarioResult {
  scenario_id: string;
  status: "running" | "completed" | "failed";
  runtime_seconds: number;
  params: ScenarioParams;
  result: {
    optimal_k: number;
    silhouette_score: number;
    inertia: number;
    n_clusters: number;
    total_dfus: number;
    cluster_sizes: Record<string, number>;
    k_selection_results: {
      k_values: number[];
      inertias: number[];
      silhouette_scores: number[];
      gap_stats: number[] | null;
    };
    profiles: ClusterProfile[];
    feature_importance: { feature: string; variance_ratio: number }[];
  } | null;
  error?: string;
}

// Component state
const [whatIfExpanded, setWhatIfExpanded] = useState(false);
const [scenarioParams, setScenarioParams] = useState<ScenarioParams>(defaults);
const [scenarioA, setScenarioA] = useState<ScenarioResult | null>(null);
const [scenarioB, setScenarioB] = useState<ScenarioResult | null>(null);
const [runningScenario, setRunningScenario] = useState(false);
const [defaults, setDefaults] = useState<ScenarioParams | null>(null);
```

### Backend State

- Scenario results stored in temp directories: `/tmp/clustering_scenario_<id>/`
- Each directory contains: `clustering_features.csv`, `cluster_assignments.csv`, `cluster_centroids.csv`, `cluster_metadata.json`, `cluster_labels.csv`, `cluster_profiles.json`
- Temp directories cleaned up after 1 hour (configurable) or on server restart
- No database writes until promote

---

## Interaction Flows

### Flow 1: First Scenario Run

1. User expands "What-If Scenarios" panel
2. UI fetches `GET /clustering/defaults` and populates all controls
3. User adjusts parameters (e.g., changes K range to 4–8, enables PCA)
4. User clicks "Run Scenario"
5. UI sends `POST /clustering/scenario` with current params
6. Loading state shown with `LoadingElement`
7. Response received — Scenario A populated
8. K-selection chart, radar chart, bar chart, and summary table rendered
9. No comparison shown (only one scenario exists)

### Flow 2: Comparison Run

1. With Scenario A already populated, user adjusts parameters
2. User clicks "Run Scenario" again
3. Response populates Scenario B
4. All charts now show both scenarios overlaid
5. Comparison table appears with deltas
6. "Promote" buttons appear for both scenarios

### Flow 3: Labeling-Only Rerun

Some parameters only affect labeling (volume/CV/seasonality/zero-demand thresholds) and do not require retraining the model. The UI detects this:

1. If only `label_params` changed (model_params and feature_params unchanged from the last run):
   - UI sends the request with a `relabel_only: true` flag
   - Backend skips feature generation and model training
   - Only re-runs `label_clusters.py` with new thresholds on the existing centroids
   - Response time: < 1 second (vs 10–120s for full run)
2. UI shows a badge: "Relabel only (instant)" on the results

### Flow 4: Promote

1. User clicks "Promote Scenario A"
2. Confirmation dialog shown with scenario summary
3. User confirms
4. `POST /clustering/scenario/<id>/promote` called
5. Backend writes to `dim_dfu.ml_cluster`
6. Success toast shown
7. Main cluster summary table refreshes
8. Promoted scenario badge changes to "Active"

---

## Backend Implementation Notes

### Scenario Runner

The backend scenario endpoint should:

1. Create a unique temp directory: `/tmp/clustering_scenario_<uuid>/`
2. Call the existing scripts as Python functions (not subprocess) where possible, with overridden output paths and parameters
3. Capture all output artifacts (CSVs, JSONs) in the temp dir
4. Return the aggregated result as JSON (no PNGs — the UI renders interactive charts)
5. Log the run to MLflow under experiment `dfu_clustering_whatif` with a `scenario_id` tag

### Concurrency

- Only one scenario can run at a time per server (clustering is CPU-intensive)
- If a second request arrives while one is running, return `409 Conflict` with estimated remaining time
- The UI disables the "Run Scenario" button while a run is in progress

### Relabel Shortcut

When `relabel_only: true`:
- Read existing centroids from the previous scenario's temp dir (or from `data/clustering/cluster_centroids.csv` if no scenario specified)
- Apply labeling logic with new thresholds
- Return updated profiles without re-running feature generation or model training

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `mvp/demand/api/main.py` | Modify | Add `/clustering/scenario`, `/clustering/scenario/<id>`, `/clustering/scenario/<id>/promote`, `/clustering/defaults` endpoints |
| `mvp/demand/frontend/src/App.tsx` | Modify | Add What-If panel to Clusters tab with parameter controls, scenario slots, charts, comparison table, promote flow |
| `mvp/demand/scripts/run_clustering_scenario.py` | Create | Scenario runner that orchestrates feature gen + training + labeling in a temp dir |
| `mvp/demand/config/clustering_config.yaml` | No change | Read by `/clustering/defaults` endpoint; not modified by scenarios |
| `docs/design-specs/feature29.md` | Create | This spec |
| `docs/design-specs/feature1.md` | Modify | Add Feature 29 to implemented features list |
| `CLAUDE.md` | Modify | Add scenario endpoints and What-If UI to relevant sections |

## Dependencies

No new npm packages required. Uses existing:
- **Recharts** — `RadarChart`, `LineChart`, `BarChart` (already in the project)
- **shadcn/ui** — `Card`, `Badge`, `Button`, slider/toggle if available (or plain `<input type="range">`)
- **Tailwind CSS** — all styling via existing semantic tokens

Backend: no new Python packages. Uses existing `scikit-learn`, `pandas`, `yaml`, `mlflow`.

## Testing & Validation

### Manual Testing Checklist

1. **Defaults load** — expanding the panel populates all controls from `clustering_config.yaml` values
2. **Validation** — setting K min > K max shows inline error, "Run" button disabled
3. **First run** — produces Scenario A with all charts and summary table
4. **Second run** — produces Scenario B; comparison table and overlaid charts appear
5. **Relabel shortcut** — changing only labeling thresholds and running completes in < 1s
6. **Promote** — confirmation dialog shows, DFUs updated, main cluster table refreshes
7. **Concurrent block** — running two scenarios simultaneously returns 409 on the second
8. **Theme support** — all charts and controls render correctly in Light, Dark, and Midnight themes
9. **Reset** — "Reset to Defaults" button restores all controls to config values
10. **Error handling** — if the pipeline fails (e.g., insufficient data), error message shown in results area

### Automated Tests

- API unit tests for `/clustering/scenario` with mock pipeline
- API unit tests for `/clustering/defaults` returning correct YAML values
- Frontend: verify parameter controls render and update state
- Frontend: verify chart data transformation from API response to Recharts props

## Performance Considerations

- **Feature generation** is the slowest step (~5–15s) due to SQL queries and joins. Consider caching the base feature matrix and only regenerating when `time_window_months` or `min_months_history` changes.
- **Gap statistic** adds ~20–40s. Default `skip_gap=true` in the UI for faster iteration, with a note explaining the trade-off.
- **Relabel shortcut** avoids the expensive steps entirely when only thresholds change.
- **Chart rendering** — Recharts handles up to 20 clusters and 20 K values without performance issues. No virtualization needed.

## Future Enhancements (Out of Scope for Feature 29)

1. **Algorithm selection** — add DBSCAN, Agglomerative, Gaussian Mixture as alternatives to KMeans
2. **Feature importance visualization** — SHAP-style feature contribution per cluster
3. **Scenario persistence** — save scenarios to database for cross-session comparison
4. **Auto-tune** — backend runs a grid of parameter combinations and returns the Pareto-optimal set
5. **Cluster stability analysis** — bootstrap resampling to show how stable cluster assignments are across random seeds
6. **Export** — download scenario results as CSV/PDF report
7. **DFU preview** — click a cluster in results to see sample DFUs and their sales time series