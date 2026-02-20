# Feature 20 — DeepAR Backtesting Implementation

## Objective

Add Amazon-style DeepAR autoregressive probabilistic forecasting to the backtesting framework. DeepAR uses LSTM-based sequence-to-sequence architecture to learn shared patterns across all DFUs, producing both point forecasts and prediction intervals. Reuses the same expanding-window timeframe structure (Feature 8), shared loader (Feature 9), and champion selection pipeline (Feature 15).

## Why DeepAR

| Capability | Tree Models (LGBM/CatBoost/XGBoost) | PatchTST | DeepAR |
|---|---|---|---|
| Learns cross-series patterns | Via categorical features | Via global training | Via global LSTM + embeddings |
| Handles intermittent demand | Moderate | Moderate | Strong (zero-inflated likelihood) |
| Cold start (new DFUs) | Needs history for lags | Needs seq_len history | Learns from similar series |
| Probabilistic output | No (point only) | No (point only) | Yes (mean + quantiles) |
| Temporal dynamics | Via hand-crafted lag/rolling features | Via patch attention | Via recurrent hidden state |
| Seasonality capture | Via calendar features | Via positional encoding | Via calendar covariates + recurrence |

DeepAR complements the existing model zoo by adding:
1. **Probabilistic forecasts** — prediction intervals for safety stock and service level planning
2. **Autoregressive generation** — each prediction conditions on all previous predictions (multi-step coherence)
3. **Categorical embeddings** — learns dense representations of DFU attributes (cluster, region, brand) directly in the network
4. **Robustness to intermittent demand** — Gaussian likelihood naturally handles zero-heavy distributions when combined with log1p transform

## Architecture

### DeepAR Model (`deepar_model.py`)

```
Input: (batch, seq_len=12) log1p qty values + (batch, n_covariates) static features
                                    ↓
                    ┌───────────────────────────────┐
                    │  Covariate Projection          │
                    │  Linear(n_cov, hidden_size)     │
                    └───────────────┬───────────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │  LSTM Encoder                   │
                    │  2 layers, hidden_size=64        │
                    │  Input: concat(qty_t, cov_proj)  │
                    │  Output: hidden states           │
                    └───────────────┬───────────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │  Distribution Head              │
                    │  Linear(hidden_size, 2)          │
                    │  Output: (mu, sigma) per step    │
                    └───────────────┬───────────────┘
                                    ↓
                    ┌───────────────────────────────┐
                    │  Gaussian Likelihood Loss        │
                    │  -log p(y | mu, sigma)           │
                    │  Point forecast: mu              │
                    └───────────────────────────────┘
```

### Key Design Decisions

1. **Single-step prediction mode** — For consistency with the backtesting framework, each target month is predicted independently (same as PatchTST). The model sees 12 months of history and predicts the next month. This avoids autoregressive error accumulation during backtest evaluation while still benefiting from the LSTM's temporal modeling.

2. **Gaussian likelihood** — Output layer produces `(mu, sigma)` parameters. Training minimizes negative log-likelihood. Point forecast is the mean `mu`. Prediction intervals available via `mu +/- z * sigma` for any confidence level.

3. **Log1p transform** — Input quantities are `log1p(max(qty, 0))` transformed (same as PatchTST). Predictions are inverse-transformed via `expm1()` and floored at 0.

4. **Covariate injection** — Static DFU/item features are projected to `hidden_size` and concatenated with the time series input at each step. This is the standard DeepAR approach for conditioning on metadata.

5. **Teacher forcing** — During training, the model receives actual historical values as input. During prediction, it uses the same actual historical context (no autoregressive rollout needed in single-step mode).

### Model Parameters

| Parameter | Default | Description |
|---|---|---|
| `hidden_size` | 64 | LSTM hidden state dimension |
| `num_layers` | 2 | Number of stacked LSTM layers |
| `dropout` | 0.1 | Dropout between LSTM layers |
| `seq_len` | 12 | Lookback context window (months) |
| `n_covariates` | 11 | Static + calendar features |

### Training Parameters

| Parameter | Default | Description |
|---|---|---|
| `epochs` | 30 | Maximum training epochs |
| `batch_size` | 512 | Training batch size |
| `learning_rate` | 1e-3 | AdamW learning rate |
| `patience` | 5 | Early stopping patience (validation NLL) |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `loss` | Gaussian NLL | Negative log-likelihood loss |
| `scheduler` | CosineAnnealingLR | Learning rate decay |

### ~40K Trainable Parameters

```
Covariate projection:  11 * 64 + 64 = 768
LSTM layer 1:          4 * (64 + 64 + 1) * 64 = 33,024
LSTM layer 2:          4 * (64 + 64 + 1) * 64 = 33,024
Distribution head:     64 * 2 + 2 = 130
─────────────────────────────────
Total:                 ~67K parameters
```

## Model IDs

| Strategy | model_id | Description |
|---|---|---|
| Global | `deepar_global` | One DeepAR for all DFUs |
| Per-cluster | `deepar_cluster` | Separate DeepAR per `ml_cluster` |
| Transfer | `deepar_transfer` | Global base → per-cluster fine-tune |

## Data Flow

Same as PatchTST (Feature 19), identical to the established deep learning backtest pattern:

```
PostgreSQL (fact_sales_monthly + dim_dfu + dim_item)
    ↓
Load sales pivot + DFU/item attributes
    ↓
Generate 10 timeframes (A-J, expanding window)
    ↓
For each timeframe:
  ├─ Mask future sales in pivot
  ├─ Build (sequence, covariate, target) tuples
  ├─ Train DeepAR (AdamW + CosineAnnealing + early stopping)
  └─ Predict → expm1 → floor at 0
    ↓
Combine predictions across timeframes
    ↓
Assign execution lag per DFU
    ↓
Deduplicate (latest timeframe wins)
    ↓
Attach actuals from fact_sales_monthly
    ↓
Output:
  ├─ backtest_predictions.csv (execution-lag only)
  ├─ backtest_predictions_all_lags.csv (lag 0-4 archive)
  └─ backtest_metadata.json
    ↓
MLflow logging (demand_backtest experiment)
    ↓
load_backtest_forecasts.py → PostgreSQL
    ↓
fact_external_forecast_monthly + backtest_lag_archive
```

## Feature Engineering

Identical to PatchTST — no hand-crafted lag/rolling features. The LSTM learns temporal patterns directly from raw sequences.

**Input per sample:**
- `sequences`: `(seq_len=12,)` — log1p monthly qty values (lookback window)
- `covariates`: `(n_cov=11,)` — static + calendar features

**Covariate vector (11 features):**

| Index | Feature | Source |
|---|---|---|
| 0 | `execution_lag` | dim_dfu |
| 1 | `total_lt` | dim_dfu |
| 2 | `ml_cluster` (label-encoded) | dim_dfu |
| 3 | `region` (label-encoded) | dim_dfu |
| 4 | `brand` (label-encoded) | dim_dfu |
| 5 | `abc_vol` (label-encoded) | dim_dfu |
| 6 | `case_weight` | dim_item |
| 7 | `item_proof` | dim_item |
| 8 | `bpc` | dim_item |
| 9 | `month_sin` | calendar |
| 10 | `month_cos` | calendar |

## Sequence Building

Reuses the same `build_sequences()` pattern from PatchTST:

```python
# Sales pivot: (n_dfus, n_months) matrix
# For each target month:
#   lookback = pivot[dfu, target_idx-12 : target_idx]
#   seq = log1p(max(lookback, 0))
#   target = log1p(max(qty_at_target, 0))
#   cov = [numeric_dfu, label_encoded_cat, numeric_item, calendar]
```

**Causal constraint:** Sequences only contain data strictly before the target month. Future sales masked at `train_end` cutoff per timeframe.

## Implementation Files

| File | Purpose |
|---|---|
| `scripts/deepar_model.py` | DeepAR model class, Dataset, train/predict, transfer helpers |
| `scripts/run_backtest_deepar.py` | Backtest script (same structure as `run_backtest_patchtst.py`) |

### `deepar_model.py` — Module Structure

```python
# Device detection (reuse pattern from PatchTST)
def get_device(override=None) -> torch.device

# Dataset
class DemandDataset(Dataset):
    """(sequences, covariates, targets) → PyTorch tensors."""

# Model
class DeepARModel(nn.Module):
    """
    LSTM encoder + Gaussian distribution head.

    Forward pass:
      1. Project covariates to hidden_size
      2. At each timestep, concatenate (qty_t, cov_proj)
      3. Feed through stacked LSTM
      4. Distribution head: hidden → (mu, sigma)
      5. Return (mu, sigma) for last timestep
    """

    def __init__(self, seq_len, hidden_size, num_layers, dropout, n_covariates):
        self.cov_proj = nn.Linear(n_covariates, hidden_size)
        self.lstm = nn.LSTM(
            input_size=hidden_size + 1,  # qty + projected covariates
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus(),  # ensures sigma > 0
        )

    def forward(self, seq, cov) -> tuple[Tensor, Tensor]:
        """Returns (mu, sigma) for next-step prediction."""
        # seq: (batch, seq_len)
        # cov: (batch, n_cov)
        cov_emb = self.cov_proj(cov)           # (batch, hidden_size)
        cov_expanded = cov_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_size)
        x = torch.cat([seq.unsqueeze(-1), cov_expanded], dim=-1)     # (batch, seq_len, hidden_size+1)
        lstm_out, _ = self.lstm(x)             # (batch, seq_len, hidden_size)
        last_hidden = lstm_out[:, -1, :]       # (batch, hidden_size)
        mu = self.mu_head(last_hidden)         # (batch, 1)
        sigma = self.sigma_head(last_hidden)   # (batch, 1)
        return mu.squeeze(-1), sigma.squeeze(-1)

# Training
def train_model(model, train_loader, val_loader, device, epochs, lr, patience, weight_decay):
    """Train with Gaussian NLL loss + AdamW + CosineAnnealing + early stopping."""
    # Loss: -log N(target | mu, sigma^2)
    # = 0.5 * log(2*pi*sigma^2) + (target - mu)^2 / (2*sigma^2)
    # Using torch.nn.GaussianNLLLoss

# Prediction
def predict_model(model, dataset, device, batch_size):
    """Returns (mu_array, sigma_array) in log1p space."""

# Transfer helpers
def freeze_for_transfer(model, freeze_layers=1)
def unfreeze_all(model)
```

### `run_backtest_deepar.py` — Script Structure

Identical structure to `run_backtest_patchtst.py`:

```python
# CLI args: --cluster-strategy, --model-id, --n-timeframes, --output-dir
# DeepAR architecture args: --hidden-size, --num-layers, --dropout, --seq-len
# Training args: --epochs, --batch-size, --learning-rate, --patience, --weight-decay
# Transfer args: --transfer-epochs, --transfer-min-rows, --transfer-freeze-layers
# Device: --device (auto: MPS > CUDA > CPU)

def main():
    # Step 1: Load data from Postgres (same SQL queries)
    # Step 2: Generate timeframes (A-J, expanding window)
    # Step 3: Build sales pivot table
    # Step 4: Train & predict per timeframe
    #   - build_sequences() for train and predict months
    #   - train_and_predict_global/per_cluster/transfer
    # Step 5: Combine, assign execution lag, deduplicate, attach actuals
    # Step 6: Save CSVs (backtest_predictions.csv, backtest_predictions_all_lags.csv)
    # Step 7: Save metadata JSON + MLflow logging
```

## Training Strategy

### Gaussian Negative Log-Likelihood Loss

```python
loss = torch.nn.GaussianNLLLoss(reduction='mean')
# loss(mu, target, sigma**2) = 0.5 * [log(sigma^2) + (target - mu)^2 / sigma^2 + log(2*pi)]
```

This is the core DeepAR training objective. Unlike MSE/Huber used by tree models and PatchTST:
- Learns both the **mean** (point forecast) and **uncertainty** (sigma)
- Penalizes overconfident wrong predictions more heavily
- Natural handling of heteroscedastic noise (different DFUs can have different variance)

### Training Loop

```python
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
criterion = GaussianNLLLoss()

for epoch in range(epochs):
    model.train()
    for seq, cov, target in train_loader:
        mu, sigma = model(seq, cov)
        loss = criterion(mu, target, sigma**2)  # variance = sigma^2
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    scheduler.step()

    # Validation for early stopping
    model.eval()
    val_loss = evaluate(model, val_loader)
    if val_loss < best_val_loss:
        best_state = deepcopy(model.state_dict())
    else:
        patience_counter += 1
        if patience_counter >= patience:
            model.load_state_dict(best_state)
            break
```

### Prediction

```python
model.eval()
with torch.no_grad():
    mu, sigma = model(seq, cov)
    # Point forecast (in original space):
    point = expm1(mu)  # inverse log1p
    point = max(point, 0)  # floor at 0

    # Prediction intervals (optional, stored in metadata):
    # PI_90 = expm1(mu +/- 1.645 * sigma)
    # PI_95 = expm1(mu +/- 1.96 * sigma)
```

## Cluster Strategies

### Global (`deepar_global`)

One DeepAR model trained on all DFUs. Cluster information is provided via the label-encoded `ml_cluster` covariate — the LSTM learns cluster-specific patterns through the covariate projection.

### Per-Cluster (`deepar_cluster`)

Separate DeepAR model per `ml_cluster`. Clusters with <50 training sequences are skipped (predictions zeroed). The `ml_cluster` covariate is excluded from features since it's the grouping key.

### Transfer (`deepar_transfer`)

Two-phase training (same as PatchTST transfer):

1. **Phase 1 (Base):** Train global DeepAR on ALL data (no `ml_cluster` covariate)
2. **Phase 2 (Fine-tune):** For each cluster with >= `transfer_min_rows` (default 20):
   - Deep-copy base model weights
   - Freeze LSTM layer 1 (`transfer_freeze_layers=1`)
   - Fine-tune on cluster data for `transfer_epochs` (default 10) at `lr * 0.1`
3. **Fallback:** Clusters < `transfer_min_rows` or unassigned DFUs use base model predictions

## Make Targets

```makefile
# DeepAR backtesting
backtest-deepar:
	$(UV) python scripts/run_backtest_deepar.py --cluster-strategy global

backtest-deepar-cluster:
	$(UV) python scripts/run_backtest_deepar.py --cluster-strategy per_cluster

backtest-deepar-transfer:
	$(UV) python scripts/run_backtest_deepar.py --cluster-strategy transfer
```

## CLI Usage

```bash
# Global DeepAR backtest (default)
make backtest-deepar

# Per-cluster DeepAR backtest
make backtest-deepar-cluster

# Transfer learning DeepAR backtest
make backtest-deepar-transfer

# Load into Postgres (shared loader)
make backtest-load

# Custom hyperparameters via CLI
uv run python scripts/run_backtest_deepar.py \
    --cluster-strategy global \
    --hidden-size 128 \
    --num-layers 3 \
    --epochs 50 \
    --batch-size 1024 \
    --learning-rate 5e-4 \
    --device mps
```

## Output Format

Identical to all other models — compatible with shared loader:

### Main CSV (`backtest_predictions.csv`)

| Column | Type | Description |
|---|---|---|
| `forecast_ck` | TEXT | `dmdunit_dmdgroup_loc_fcstdate_startdate` |
| `dmdunit` | TEXT | Item |
| `dmdgroup` | TEXT | Product group |
| `loc` | TEXT | Location |
| `fcstdate` | DATE | Forecast origin (startdate - lag months) |
| `startdate` | DATE | Target month |
| `lag` | INT | Execution lag for this DFU |
| `execution_lag` | INT | DFU's operational lead time |
| `basefcst_pref` | NUMERIC | Point forecast (mu, in original space) |
| `tothist_dmd` | NUMERIC | Actual demand |
| `model_id` | TEXT | `deepar_global` / `deepar_cluster` / `deepar_transfer` |

### Archive CSV (`backtest_predictions_all_lags.csv`)

Same as main CSV + `timeframe` column (A-J).

### Metadata JSON (`backtest_metadata.json`)

```json
{
  "model_id": "deepar_global",
  "cluster_strategy": "global",
  "n_timeframes": 10,
  "deepar_params": {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "seq_len": 12,
    "n_covariates": 11
  },
  "training_params": {
    "epochs": 30,
    "batch_size": 512,
    "learning_rate": 0.001,
    "patience": 5,
    "weight_decay": 0.0001,
    "loss": "gaussian_nll",
    "device": "mps"
  },
  "n_predictions": 54320,
  "n_dfus": 5432,
  "date_range": {
    "earliest": "2023-02-01",
    "latest": "2026-01-01"
  },
  "timeframes": [...],
  "accuracy_at_execution_lag": {
    "n_rows": 54320,
    "wape": 32.1,
    "bias": 0.015,
    "accuracy_pct": 67.9
  }
}
```

## Dependencies

- `torch>=2.0.0` (already in `pyproject.toml`)
- No additional Python packages required

DeepAR is implemented as a custom PyTorch module (`deepar_model.py`), avoiding heavy dependencies like GluonTS or PyTorch Forecasting. This keeps the implementation lightweight, consistent with the PatchTST approach, and gives full control over the training loop.

## Device Support

| Platform | Device | Detection |
|---|---|---|
| macOS (Apple Silicon) | MPS | `torch.backends.mps.is_available()` |
| Linux (NVIDIA) | CUDA | `torch.cuda.is_available()` |
| Any | CPU | Fallback |

Same auto-detection as PatchTST via `get_device()`.

## DeepAR vs PatchTST: When to Use Each

| Scenario | Better Model | Why |
|---|---|---|
| Smooth, trending demand | PatchTST | Attention captures long-range trends |
| Intermittent / sparse demand | DeepAR | Gaussian likelihood handles zero-heavy data |
| Need prediction intervals | DeepAR | Probabilistic output (mu + sigma) |
| Short history (<12 months) | DeepAR | LSTM more forgiving with limited context |
| Strong seasonality | Either | Both capture via calendar covariates |
| Many related series | DeepAR | Cross-series learning via shared LSTM weights |

Champion selection (Feature 15) automatically picks the best model per DFU, so both can coexist and complement each other.

## Champion Selection Integration

Add DeepAR models to `config/model_competition.yaml`:

```yaml
competition:
  models:
  - lgbm_global
  - lgbm_cluster
  - lgbm_transfer
  - catboost_global
  - catboost_cluster
  - xgboost_global
  - xgboost_cluster
  - patchtst_global
  - deepar_global
  - deepar_cluster
  - deepar_transfer
```

No code changes needed — champion selection is model-agnostic and works with any `model_id` in the forecast table.

## MLflow Logging

Logged under experiment `demand_backtest` (shared with all models):

```python
mlflow.set_tag("model_type", "deepar_backtest")
mlflow.set_tag("cluster_strategy", args.cluster_strategy)
mlflow.set_tag("model_id", model_id)
mlflow.set_tag("device", str(device))
mlflow.log_params({**deepar_params, **training_params})
mlflow.log_metrics({"n_predictions": ..., "wape": ..., "accuracy_pct": ..., "bias": ...})
mlflow.log_artifact(output_path)
mlflow.log_artifact(archive_path)
mlflow.log_artifact(meta_path)
```

## Verification

```bash
# 1. Run DeepAR backtest
make backtest-deepar

# 2. Check output files
ls -lh mvp/demand/data/backtest/backtest_predictions.csv
ls -lh mvp/demand/data/backtest/backtest_predictions_all_lags.csv
cat mvp/demand/data/backtest/backtest_metadata.json | python3 -m json.tool

# 3. Load into Postgres
make backtest-load

# 4. Verify in database
docker exec demand-mvp-postgres psql -U demand -d demand_mvp \
  -c "SELECT model_id, COUNT(*) FROM fact_external_forecast_monthly WHERE model_id LIKE 'deepar%' GROUP BY 1"

docker exec demand-mvp-postgres psql -U demand -d demand_mvp \
  -c "SELECT model_id, lag, COUNT(*) FROM backtest_lag_archive WHERE model_id LIKE 'deepar%' GROUP BY 1,2 ORDER BY 1,2"

# 5. Check accuracy in UI
# Open http://127.0.0.1:5173 → Forecast → Model selector → deepar_global

# 6. Run champion selection with DeepAR included
make champion-select
```

## Files Modified

| File | Change |
|---|---|
| `scripts/deepar_model.py` | **NEW** — DeepAR model class, Dataset, train/predict, transfer helpers |
| `scripts/run_backtest_deepar.py` | **NEW** — Backtest script (same structure as PatchTST) |
| `Makefile` | Add `backtest-deepar`, `backtest-deepar-cluster`, `backtest-deepar-transfer` targets |
| `config/model_competition.yaml` | Add `deepar_global`, `deepar_cluster`, `deepar_transfer` to competing models |
| `docs/design-specs/feature20.md` | **NEW** — This design spec |
| `docs/design-specs/feature1.md` | Add Feature 20 to implemented features list |
| `CLAUDE.md` | Add DeepAR entries to Key Files, Common Commands, Design Specs |
| `mvp/demand/docs/ARCHITECTURE.md` | Add DeepAR to ML Pipeline Components |
| `mvp/demand/docs/README.md` | Add DeepAR Backtesting section |
| `mvp/demand/docs/RUNBOOK.md` | Add DeepAR backtest instructions |

## Dependencies on Other Features

| Feature | Dependency |
|---|---|
| Feature 7 (Clustering) | `ml_cluster` assignments in `dim_dfu` |
| Feature 8 (Backtesting Framework) | Expanding window timeframe structure |
| Feature 9 (LGBM / Shared Loader) | `load_backtest_forecasts.py` for Postgres loading |
| Feature 15 (Champion Selection) | Competes in model selection via YAML config |
| Feature 17 (DFU Analysis) | Auto-appears in multi-model overlay chart |

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| LSTM slower to train than tree models | MPS/CUDA GPU acceleration; early stopping limits wasted epochs |
| Gaussian assumption may not fit all DFUs | Log1p transform normalizes; champion selection filters poor performers |
| Overfitting on small clusters | Transfer learning fallback; min_rows threshold |
| Memory pressure with large batches | Default batch_size=512 (same as PatchTST); DataLoader num_workers=0 for MPS safety |
