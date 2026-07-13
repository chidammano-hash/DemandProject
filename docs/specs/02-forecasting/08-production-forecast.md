# Production Forecast Pipeline

> Trains ML models on full sales history up to the planning date, then generates forward-looking point and probabilistic forecasts for the next 24 months. This is the final step that turns backtest-validated algorithms into actionable demand plans.

| | |
|---|---|
| **Status** | Implemented |
| **UI Tab** | Inv. Planning (Demand Forecast panel), Model Experimentation Studio |
| **Key Files** | `scripts/ml/train_production_models.py`, `scripts/forecasting/generate_production_forecasts.py`, `api/routers/forecasting/production_forecast.py`, `config/forecasting/forecast_pipeline_config.yaml`, `sql/039_create_production_forecast.sql`, `frontend/src/tabs/inv-planning/DemandForecastPanel.tsx` |

---

## Problem

Backtesting evaluates model accuracy on historical hold-out windows, but those
evaluation fits are not safe production artifacts. Planners need a dedicated
pipeline that (1) final-refits LightGBM, N-HiTS, and N-BEATS on all closed
history with immutable lineage, (2) routes MSTL and Chronos 2E through the same
direct adapters used in backtesting, (3) generates forward-looking predictions,
and (4) attaches confidence interval bands derived from backtest residuals.

The base-model roster is deliberately fixed to exactly five algorithms:
LightGBM (`lgbm_cluster`), N-HiTS (`nhits`), N-BEATS (`nbeats`), MSTL (`mstl`),
and Chronos 2E (`chronos2_enriched`). `external`, `champion`, `ensemble`, and
`ceiling` are comparison or routing identities, not additional base models.

## Production Forecast vs. Backtest

| Aspect | Backtest | Production Forecast |
|--------|----------|---------------------|
| **Purpose** | Evaluate accuracy on known actuals | Generate actionable future predictions |
| **Training window** | Expanding or sliding window, partial history | Full history up to planning date |
| **Prediction target** | Historical months where actuals exist | T+1 through T+24 (future months) |
| **Evaluation** | Compare predictions to actuals (WAPE, accuracy) | No evaluation possible -- predictions are the output |
| **Artifacts** | Evaluation evidence | Atomic LightGBM bundle + immutable N-HiTS/N-BEATS artifacts; MSTL/Chronos direct |
| **Frequency** | On-demand or after data refresh | Monthly, on the 2nd after sales close |

## Pipeline Steps

The production forecast pipeline has two major phases: **Train** and **Generate**.

### Step 1: Train Production Models

Script: `scripts/ml/train_production_models.py`


1. Validate the synchronized `fact_sales_monthly_original` mirror and load it through the latest fully closed month (the month before the planning month; partial planning-month actuals are excluded)
2. Build the full feature matrix using `build_feature_matrix()` -- identical features to backtest but with all available months
3. Split into train/validation sets: the last `val_fraction` (20%) of months serve as early-stopping validation
4. Train one LightGBM model per promoted cluster (or one explicitly declared global model when clustering is disabled) and global N-HiTS/N-BEATS final fits
5. Stamp source batch/hash, history end, configuration, cluster lineage, and generator contract
6. Publish the complete LightGBM set under `data/models/lgbm_cluster/production_tree/versions/<artifact_set_id>/`, verify every file and checksum, then atomically replace `data/models/lgbm_cluster/production_tree/active.json`; a partial failure leaves the previous active version untouched
7. Fail closed when any required cluster artifact is missing; production never substitutes an unrelated cluster model

There is no fallback from the immutable-original mirror to
`fact_sales_monthly`. The latest positive completed sales audit must represent a
canonical dual-track reload (not legacy `safe_upsert`), its output row count
must equal the mirror, and `MAX(fact_sales_monthly_original.load_ts)` must be at
least the batch `started_at`. Run `make normalize-sales && make load-sales` once
to repair missing or stale mirror lineage.

Usage:
```bash
# Train a single model
make train-production MODEL=lgbm_cluster

# Train every persisted production model (LightGBM, N-BEATS, N-HiTS)
make train-production-all
```

### Step 2: Generate Point Forecasts

Script: `scripts/forecasting/generate_production_forecasts.py`

For each DFU:

1. Look up the champion model assignment (from champion selection) or use `fallback_model_id`
2. Route LightGBM/N-HiTS/N-BEATS winners through validated artifacts and MSTL/Chronos winners through their canonical direct adapters
3. Generate T+1 through T+24 without using future actuals
4. Write candidate rows to `fact_production_forecast_staging` with generation lineage
5. Promote the validated staging generation into `fact_production_forecast`


**Fail loud on prediction failure (no silent zero-fill or mislabeled fallback):** if LightGBM prediction fails, the run logs the full traceback and re-raises; it does not substitute zeros. `common/ml/production_non_tree.py` likewise requires the real MSTL, N-HiTS, N-BEATS, or Chronos 2E adapter to return the complete requested DFU-month grid with the correct model id and finite nonnegative values. Missing optional dependencies, empty results, or partial results abort generation instead of substituting a heuristic under the canonical model label.

Every persisted run is stamped with
`generator_contract_version=canonical-five-artifact-lineage-v2`. Promotion and
snapshot retry require that exact contract, so staging created by the retired
heuristic dispatcher cannot be reused as a canonical-model forecast. Chronos 2E
also rejects NaN/Inf output at the adapter boundary; it never converts invalid
model output to zero.

The generation manifest records the exact persisted versions used:
`tree_artifacts.lgbm_cluster.artifact_set_id` for LightGBM and each neural
artifact id under `neural_artifacts`. A snapshot contender therefore remains
bound to the same final-fit evidence as its immutable staging payload.

The production `Dockerfile` installs `libgomp1` and the `foundation`, `dl`, and
`statistical` extras. A local base install must run
`uv sync --extra foundation --extra dl --extra statistical` before exercising
all five models.

**Streaming reads:** Both `scripts/forecasting/generate_production_forecasts.py`
and `scripts/ml/train_meta_learner.py` load their large sales / backtest frames
via `read_sql_chunked()` from `common/core/sql_helpers.py` (chunk size
`DEFAULT_CHUNK_SIZE`). This avoids materializing multi-million-row result sets
in memory and keeps RSS bounded during long-horizon production runs.

### Step 3: Generate Probabilistic Forecasts (Confidence Intervals)

After point forecasts are written:

1. Load backtest residuals from `backtest_lag_archive` for the champion model's algorithm
2. Compute per-DFU RMSE (sigma) from forecast-minus-actual residuals
3. Apply the three-level fallback hierarchy: DFU-level sigma (if >= 6 residual observations) > cluster-level sigma > global sigma
4. Apply guard rails: `sigma_floor` (1.0 units minimum) and `sigma_cap_multiplier` (3x global median)
5. Scale sigma by horizon using `horizon_scaling` mode (default `sqrt` -- uncertainty grows like a random walk)
6. Compute bounds: `lower = max(0, forecast - z_lower * sigma * scale)`, `upper = forecast + z_upper * sigma * scale`
7. Update `forecast_qty_lower` and `forecast_qty_upper` columns in `fact_production_forecast`

See [Forecast CI Bands](./10-forecast-ci-bands.md) for full details on the residual-based CI methodology.

### Step 4: Champion Routing

When `model_selection.strategy` is `champion` (default):

1. Champion selection assigns each DFU-month a single model or an evaluated ensemble based only on causal prior evidence
2. LightGBM, N-HiTS, and N-BEATS must have current immutable final-fit artifacts; MSTL and Chronos 2E do not persist a product-specific fit
3. LightGBM DFUs load the exact promoted-cluster bundle, neural DFUs load their lineage-validated global artifact, and MSTL/Chronos DFUs use `common/ml/production_non_tree.py` with the same adapters as backtesting
4. Future-dated routes are never used; if no valid latest-as-of assignment exists, `fallback_model_id` (default `lgbm_cluster`) is used explicitly

## Model Types

### Tree Model (LightGBM)

- **Require explicit training** on full history before inference
- Produce an immutable, checksummed all-cluster artifact bundle
- Support per-cluster tuning profiles and early stopping
- Use recursive inference for multi-step horizons (lag features from prior predictions)

### Foundation Model (Chronos 2 Enriched)

- **No separate production training step required** -- it uses pretrained weights; see
  [Chronos 2 Enriched](./18-chronos-foundation-models.md)
- Consume raw sales history directly at inference time
- Output predictions for all horizons in a single forward pass
- No `.pkl` artifacts to manage

### Deep Learning Models (N-BEATS, N-HiTS)

- **Require explicit global final-refit training** through Step 1
- Publish immutable NeuralForecast artifacts bound to sales/config/history lineage
- `forecast: true` and `compete: true` in the algorithm roster, so a DFU whose champion is
  `nbeats`/`nhits` is eligible for production forecasting through `common/ml/neural_forecast.py`

### Statistical Model (MSTL)

- **No training step required** -- computed directly from history
- Uses the same `common/ml/mstl.py` adapter in backtest and production
- Participates in the same champion competition as the other four canonical models

## Artifact Management

### Directory and metadata contract

Each persisted family uses content-addressed versions plus an atomically
replaced active pointer. LightGBM versions contain the complete exact cluster
set; neural versions contain the saved global NeuralForecast model. Metadata
includes source sales batch/hash, latest closed `history_end`, model/config
checksum, generator contract, and cluster experiment for per-cluster LightGBM.

```text
data/models/lgbm_cluster/production_tree/
├── active.json                         # artifact_set_id + checksums manifest hash
└── versions/<artifact_set_id>/
    ├── metadata.json                   # exact cluster roster, config and lineage
    ├── training_metadata.json          # final-fit diagnostics for this version
    ├── checksums.json                  # complete version file manifest
    └── models/0000.pkl, 0001.pkl, ...  # opaque files mapped by metadata.json

data/models/{nhits,nbeats}/neuralforecast/
├── active.json
└── versions/<artifact_id>/             # metadata, checksums and saved global model
```

There is no supported loose `cluster_<id>.pkl`,
`data/models/lgbm_cluster/<cluster>/model.pkl`, or top-level
`training_metadata.json` lookup. For a per-cluster set, `metadata.json` must
declare exactly the currently promoted cluster labels; the explicit global
strategy must contain exactly the `global` label. Extra, missing, stale, or
checksum-mismatched files fail validation.

The filesystem active pointer is the runtime registry. The loader verifies the
pointer, version/file roster, checksums, exact model and cluster identity, and
current sales/cluster/config lineage **before** deserializing or serving any
model. Final-fit publication writes a temporary version and switches
`active.json` only after complete validation, so a failed refit cannot expose a
partially updated cluster set.

## Data Model

### `fact_production_forecast`

| Column | Type | Description |
|--------|------|-------------|
| `plan_version` | VARCHAR(30) | Version label (e.g., "2026-04") |
| `item_id` | VARCHAR(50) | Item identifier |
| `loc` | VARCHAR(50) | Location code |
| `forecast_month` | DATE | Future month being forecast |
| `forecast_qty` | NUMERIC(12,2) | Point forecast |
| `forecast_qty_lower` | NUMERIC(12,2) | Lower CI bound (P10) |
| `forecast_qty_upper` | NUMERIC(12,2) | Upper CI bound (P90) |
| `model_id` | VARCHAR(100) | Algorithm that produced this row |
| `cluster_id` | INTEGER | ml_cluster used for inference |
| `horizon_months` | SMALLINT | 1=T+1, 2=T+2, ... 24=T+24 |
| `is_recursive` | BOOLEAN | Whether recursive inference was used |
| `lag_source` | VARCHAR(20) | "actual" (T+1) or "predicted" (T+2+) |
| `run_id` | UUID | Ties rows to a single inference run |

**Grain:** `(plan_version, item_id, loc, forecast_month)`

## Cold-Start Routing

Production routing derives history through the latest closed month at full
`(item_id, customer_group, loc)` source grain. An output `(item_id, loc)` is
eligible only when **every active customer group** satisfies the applicable
history floor; a partial customer-group aggregate is never published.

The shared routing rules are:

| History Length | Routing | Rationale |
|----------------|---------|-----------|
| >= `min_history_months` (12) | Assigned canonical champion, subject to that model's stricter floor | Enough data for normal routing; MSTL still requires its configured 25 months |
| 3–11 months | Explicit LightGBM cold-start route | Keeps the eligible cohort complete without sending short histories to neural/MSTL adapters |
| < `cold_start_min_months` (3) | Skipped entirely | Too little data for any meaningful forecast |

Configured in `config/forecasting/forecast_pipeline_config.yaml` under `production_forecast`.

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/forecast/production` | Promoted forecast rows for a specific DFU + plan version (future) |
| GET | `/forecast/production/summary` | Aggregate forecast by ABC class for a plan version |
| GET | `/forecast/production/versions` | List available plan versions with metadata |
| GET | `/forecast/production/staging` | All **staged** (pre-promotion, future) forecasts for a DFU, grouped by `model_id` |
| GET | `/forecast/candidate` | All **backtest** (past, out-of-sample) predictions for a DFU, grouped by `model_id` — see [24-candidate-forecast-promotion.md](24-candidate-forecast-promotion.md) §5.4 |
| POST | `/backtest-management/{model_id}/generate` | Submit a generate job for one model → staging. Accepts `horizon` + `confidence_intervals` query params (both optional → pipeline-config defaults) |

### Generate controls (Model Tuning → Forecast)

The `ForecastPanel` threads its **Horizon** input and **Include Confidence Intervals**
toggle through to the job for every generate path (single model, champion, and
**Generate All**). The single-model `POST /{model_id}/generate` carries them as
`?horizon=&confidence_intervals=` query params; the job handler maps them to the
script's `--horizon` and `--confidence-intervals` / `--no-confidence-intervals`
flags. The CLI flag overrides `confidence_interval.enabled` in the pipeline config,
so the UI toggle actually takes effect (previously it was silently dropped for
single-model generation). A **Generate All** button submits a generate job for
every ready model (artifact-backed LightGBM/N-HiTS/N-BEATS plus direct MSTL and Chronos 2E) in one
click. Promote/generate failures (the WAPE/coverage gate's 409, or 400 for no
staged rows) surface as toasts rather than failing silently.

> **Change note (2026-06-20):** the CI batch call site now gates band generation on the **resolved** `ci_enabled` flag, not the raw `confidence_interval.enabled` config value. Passing `--confidence-intervals` (the UI "Include Confidence Intervals" toggle) had been silently ignored whenever the config had `confidence_interval.enabled: false`, producing point-only forecasts with no error. The CLI flag now reliably wins over the config default.

## Pipeline Targets

| Target | Description |
|--------|-------------|
| `make train-production MODEL=<id>` | Train a single model on full history |
| `make train-production-all` | Final-refit every persisted production family: LightGBM, N-BEATS, and N-HiTS |
| `make forecast-generate` | Generate forecasts for all DFUs (requires trained models) |
| `make forecast-generate-dfu ITEM=X LOC=Y` | Single DFU inference |
| `make forecast-generate-dry` | Preview without writing |
| `make forecast-full` | Full pipeline: train all models + generate forecasts |
| `make forecast-model MODEL=<id>` | Train + generate for a single model |
| `make forecast-prod-schema` | Create tables (one-time) |
| `make forecast-prod-all` | Schema + generate |

**Scheduler:** Runs as `generate_production_forecast` job type on the 2nd of each month at 06:00 UTC (after sales close).

## Configuration

All production forecast settings live in `config/forecasting/forecast_pipeline_config.yaml`.

### Production Training

```yaml
production_forecast:
  production_training:
    enabled: true               # Master switch for production training step
    output_dir: data/models     # Root for immutable versioned artifacts
    val_fraction: 0.20          # Last 20% of months used for validation
    min_cluster_rows: 50        # Minimum rows required; missing clusters fail the complete set
    save_metadata: true         # Save diagnostics inside each immutable version
```

### Inference

```yaml
production_forecast:
  horizon_months: 24            # Months ahead to forecast
  recursive: true               # Use recursive inference for multi-step horizons
  fallback_model_id: lgbm_cluster
```

### Confidence Intervals

```yaml
production_forecast:
  confidence_interval:
    enabled: true               # Enable probabilistic forecasts (CI bands)
    z_lower: 1.282              # Z-score for 10th percentile (P10)
    z_upper: 1.282              # Z-score for 90th percentile (P90)
    horizon_scaling: sqrt       # Scale sigma by sqrt(h) for longer horizons
    sigma_floor: 1.0            # Minimum sigma in units
    sigma_cap_multiplier: 3.0   # Cap sigma at 3x global median
```

Champion strategy and model roster live in the top-level `champion:` block;
generation routing fallbacks live in `production_forecast:`. There is no
separate `inference:` or `model_selection:` block.

## UI Integration

The **Model Experimentation Studio** provides a Train then Generate workflow:

1. Select models to train (or "all")
2. Monitor training progress (cluster-by-cluster)
3. Review training metadata (row counts, validation RMSE, feature counts)
4. Trigger forecast generation
5. View point forecasts with CI bands in the Demand Forecast panel

The monthly publish workflow also prepares exactly three WAPE-ranked
`snapshot_contender` runs. When the next release replaces the current one,
promotion archives the outgoing `champion` plus those three frozen contenders
for snapshot lags 0 through 5; no other staged forecast is part of the bounded
live-FVA archive.

## Dependencies

- [Backtest Framework](./03-backtest-framework.md) -- provides backtest residuals for CI computation
- [Champion Selection](./07-champion-selection.md) -- determines which model to use per DFU
- [Forecast CI Bands](./10-forecast-ci-bands.md) -- details the residual-based CI methodology
- Clustering (in `03-demand-intelligence/`) -- routes DFUs to correct cluster model

## See Also

- [Bias Correction](./09-bias-correction.md) -- consumes production forecasts for projection
- [Forecast CI Bands](./10-forecast-ci-bands.md) -- companion feature for uncertainty quantification
