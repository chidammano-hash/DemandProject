"""
Run PatchTST backtesting with expanding-window timeframes.

PatchTST (Patch Time Series Transformer) segments monthly demand sequences into
overlapping patches, processes them through a Transformer encoder conditioned on
static covariates, and produces a single next-month forecast.  Unlike tree-based
models, PatchTST operates in log1p space on raw quantity sequences with minimal
feature engineering.

Supports three strategies:
  - global:      One PatchTST trained on all DFUs           (model_id=patchtst_global)
  - per_cluster: Separate PatchTST per ml_cluster           (model_id=patchtst_cluster)
  - transfer:    Global base -> per-cluster fine-tune        (model_id=patchtst_transfer)

Produces two CSVs:
  - backtest_predictions.csv: execution-lag only (for fact_external_forecast_monthly)
  - backtest_predictions_all_lags.csv: lag 0-4 archive (for backtest_lag_archive)
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.backtest_framework import (
    generate_timeframes,
    load_backtest_data,
    postprocess_predictions,
    save_backtest_output,
)
from common.constants import MIN_CLUSTER_ROWS
from common.db import get_db_params
from common.mlflow_utils import log_backtest_run

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ── Sequence & covariate builders ────────────────────────────────────────────


def _build_sales_pivot(
    sales_df: pd.DataFrame,
    all_months: list[pd.Timestamp],
) -> pd.DataFrame:
    """Build a (dfu_ck x month) pivot table of qty values.

    Missing months are filled with 0.

    Returns:
        DataFrame with dfu_ck as index and months as columns.
    """
    pivot = sales_df.pivot_table(
        index="dfu_ck",
        columns="startdate",
        values="qty",
        aggfunc="sum",
        fill_value=0,
    )
    # Ensure all months present
    for m in all_months:
        if m not in pivot.columns:
            pivot[m] = 0
    pivot = pivot[sorted(pivot.columns)]
    return pivot


def _encode_categoricals(
    dfu_attrs: pd.DataFrame,
) -> tuple[dict[str, LabelEncoder], pd.DataFrame]:
    """Label-encode categorical columns in dfu_attrs.

    Encodes: ml_cluster, region, brand, abc_vol.
    Unknown / NaN values are assigned code 0 (fit on fillna("__unknown__")).

    Returns:
        (encoders dict, dfu_attrs with encoded columns appended as *_enc).
    """
    cat_cols = ["ml_cluster", "region", "brand", "abc_vol"]
    encoders: dict[str, LabelEncoder] = {}
    result = dfu_attrs.copy()

    for col in cat_cols:
        le = LabelEncoder()
        filled = result[col].fillna("__unknown__").astype(str)
        le.fit(filled)
        result[f"{col}_enc"] = le.transform(filled)
        encoders[col] = le

    return encoders, result


def _build_sequences_for_timeframe(
    sales_pivot: pd.DataFrame,
    dfu_attrs_enc: pd.DataFrame,
    item_attrs: pd.DataFrame,
    predict_months: list[pd.Timestamp],
    all_months_sorted: list[pd.Timestamp],
    train_end: pd.Timestamp,
    seq_len: int = 12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Build (sequences, covariates, targets) arrays for one timeframe.

    For each DFU and each target month in predict_months:
      - sequence:   last `seq_len` months of qty before target, log1p, left-zero-padded
      - covariates: [execution_lag, total_lt, case_weight, item_proof, bpc,
                     month_sin, month_cos, ml_cluster_enc, region_enc, brand_enc, abc_vol_enc]
      - target:     log1p(qty at target month)

    The sequence uses only months <= train_end to avoid future leakage.
    Predictions are built for ALL DFUs across ALL predict months.

    Returns:
        (sequences, covariates, targets, metadata_rows)
        where metadata_rows is a list of dicts with dfu_ck, dmdunit, dmdgroup, loc, startdate.
    """
    # Pre-compute month index lookup
    month_to_idx = {m: i for i, m in enumerate(all_months_sorted)}

    # Build DFU attribute lookup
    dfu_lookup = dfu_attrs_enc.set_index("dfu_ck")

    # Build item attribute lookup
    if len(item_attrs) > 0:
        item_lookup = item_attrs.set_index("dmdunit")
    else:
        item_lookup = pd.DataFrame()

    # Pre-allocate lists
    sequences_list = []
    covariates_list = []
    targets_list = []
    metadata_rows = []

    # All DFU CKs present in the pivot (DFUs with sales data)
    all_dfu_cks = sales_pivot.index.values
    pivot_values = sales_pivot.values  # (n_dfus, n_months)
    pivot_columns = list(sales_pivot.columns)
    col_to_idx = {m: i for i, m in enumerate(pivot_columns)}

    # Find the index of train_end in all_months for masking
    train_months_mask = [m for m in all_months_sorted if m <= train_end]

    for dfu_ck in all_dfu_cks:
        # Look up DFU attributes
        if dfu_ck not in dfu_lookup.index:
            continue
        dfu_row = dfu_lookup.loc[dfu_ck]
        dmdunit = dfu_row["dmdunit"]
        dmdgroup = dfu_row["dmdgroup"]
        loc = dfu_row["loc"]

        # DFU-level numeric attributes (with safe defaults)
        execution_lag = float(pd.to_numeric(dfu_row.get("execution_lag", 0), errors="coerce") or 0)
        total_lt = float(pd.to_numeric(dfu_row.get("total_lt", 0), errors="coerce") or 0)

        # Item-level numeric attributes
        case_weight = 0.0
        item_proof = 0.0
        bpc = 0.0
        if len(item_lookup) > 0 and dmdunit in item_lookup.index:
            item_row = item_lookup.loc[dmdunit]
            if isinstance(item_row, pd.DataFrame):
                item_row = item_row.iloc[0]
            case_weight = float(pd.to_numeric(item_row.get("case_weight", 0), errors="coerce") or 0)
            item_proof = float(pd.to_numeric(item_row.get("item_proof", 0), errors="coerce") or 0)
            bpc = float(pd.to_numeric(item_row.get("bpc", 0), errors="coerce") or 0)

        # Encoded categoricals
        ml_cluster_enc = float(dfu_row.get("ml_cluster_enc", 0))
        region_enc = float(dfu_row.get("region_enc", 0))
        brand_enc = float(dfu_row.get("brand_enc", 0))
        abc_vol_enc = float(dfu_row.get("abc_vol_enc", 0))

        # Get the pivot row index for this DFU
        dfu_pivot_idx = sales_pivot.index.get_loc(dfu_ck)

        for target_month in predict_months:
            # Build lookback sequence: last seq_len months BEFORE target month,
            # but only up to train_end
            target_idx = month_to_idx.get(target_month)
            if target_idx is None:
                continue

            # Gather lookback months: months strictly before target_month and <= train_end
            lookback_months = [
                m for m in train_months_mask
                if m < target_month
            ]
            # Take the last seq_len months
            lookback_months = lookback_months[-seq_len:]

            # Build sequence (left-zero-padded if < seq_len)
            seq = np.zeros(seq_len, dtype=np.float32)
            offset = seq_len - len(lookback_months)
            for j, m in enumerate(lookback_months):
                cidx = col_to_idx.get(m)
                if cidx is not None:
                    seq[offset + j] = pivot_values[dfu_pivot_idx, cidx]

            # Apply log1p transform
            seq = np.log1p(np.maximum(seq, 0))

            # Calendar features for target month
            month_num = target_month.month
            month_sin = math.sin(2 * math.pi * month_num / 12)
            month_cos = math.cos(2 * math.pi * month_num / 12)

            # Covariates: 11 features
            cov = np.array([
                execution_lag,
                total_lt,
                case_weight,
                item_proof,
                bpc,
                month_sin,
                month_cos,
                ml_cluster_enc,
                region_enc,
                brand_enc,
                abc_vol_enc,
            ], dtype=np.float32)

            # Target: log1p(qty at target month)
            target_cidx = col_to_idx.get(target_month)
            if target_cidx is not None:
                target_val = np.log1p(max(float(pivot_values[dfu_pivot_idx, target_cidx]), 0))
            else:
                target_val = 0.0

            sequences_list.append(seq)
            covariates_list.append(cov)
            targets_list.append(target_val)
            metadata_rows.append({
                "dfu_ck": dfu_ck,
                "dmdunit": dmdunit,
                "dmdgroup": dmdgroup,
                "loc": loc,
                "startdate": target_month,
            })

    if not sequences_list:
        return (
            np.zeros((0, seq_len), dtype=np.float32),
            np.zeros((0, 11), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            [],
        )

    return (
        np.stack(sequences_list),
        np.stack(covariates_list),
        np.array(targets_list, dtype=np.float32),
        metadata_rows,
    )


# ── Strategy runners ─────────────────────────────────────────────────────────


def _train_and_predict_global(
    sequences: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    metadata_rows: list[dict],
    device,
    arch_params: dict,
    train_params: dict,
) -> list[dict]:
    """Train one global PatchTST and predict all DFUs."""
    import torch
    from torch.utils.data import DataLoader, random_split

    from scripts.patchtst_model import (
        DemandDataset,
        PatchTSTModel,
        predict_model,
        train_model,
    )

    n_total = len(sequences)
    if n_total == 0:
        return []

    # 80/20 train/val split
    dataset = DemandDataset(sequences, covariates, targets)
    n_val = max(int(n_total * 0.2), 1)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=train_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_params["batch_size"], shuffle=False)

    model = PatchTSTModel(**arch_params)
    history = train_model(
        model, train_loader, val_loader, device,
        epochs=train_params["epochs"],
        lr=train_params["lr"],
        patience=train_params["patience"],
        weight_decay=train_params["weight_decay"],
    )
    print(f"      Training: {len(history['train_loss'])} epochs, "
          f"best_epoch={history['best_epoch']}, "
          f"best_val_loss={min(history['val_loss']):.4f}")

    # Predict on ALL sequences (not just train split)
    full_dataset = DemandDataset(sequences, covariates, targets=None)
    preds_log1p = predict_model(model, full_dataset, device, batch_size=train_params["batch_size"])

    # Convert from log1p space -> raw qty, floor at 0
    preds_raw = np.expm1(preds_log1p)
    preds_raw = np.maximum(preds_raw, 0)

    # Build prediction rows
    results = []
    for i, meta in enumerate(metadata_rows):
        results.append({
            **meta,
            "basefcst_pref": float(preds_raw[i]),
        })

    return results


def _train_and_predict_per_cluster(
    sequences: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    metadata_rows: list[dict],
    dfu_attrs_enc: pd.DataFrame,
    device,
    arch_params: dict,
    train_params: dict,
    min_cluster_rows: int = MIN_CLUSTER_ROWS,
) -> list[dict]:
    """Train separate PatchTST per ml_cluster."""
    import torch
    from torch.utils.data import DataLoader, random_split

    from scripts.patchtst_model import (
        DemandDataset,
        PatchTSTModel,
        predict_model,
        train_model,
    )

    # Build cluster assignment per sequence row
    dfu_cluster = dfu_attrs_enc.set_index("dfu_ck")["ml_cluster"].to_dict()
    row_clusters = [dfu_cluster.get(m["dfu_ck"], "__unknown__") for m in metadata_rows]
    row_clusters = [c if c is not None and str(c) != "nan" else "__unknown__" for c in row_clusters]

    clusters = sorted(set(c for c in row_clusters if c != "__unknown__"))
    all_results = []
    indices_by_cluster: dict[str, list[int]] = {}

    for i, c in enumerate(row_clusters):
        indices_by_cluster.setdefault(c, []).append(i)

    for ci, cluster_label in enumerate(clusters, 1):
        idxs = indices_by_cluster.get(cluster_label, [])
        if len(idxs) < min_cluster_rows:
            print(f"      Cluster {ci}/{len(clusters)} '{cluster_label}': "
                  f"skipped ({len(idxs)} < {min_cluster_rows} sequences), zeroing predictions")
            for idx in idxs:
                all_results.append({**metadata_rows[idx], "basefcst_pref": 0.0})
            continue

        c_seqs = sequences[idxs]
        c_covs = covariates[idxs]
        c_tgts = targets[idxs]

        # 80/20 split
        dataset = DemandDataset(c_seqs, c_covs, c_tgts)
        n_val = max(int(len(idxs) * 0.2), 1)
        n_train = len(idxs) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])

        train_loader = DataLoader(train_ds, batch_size=train_params["batch_size"], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=train_params["batch_size"], shuffle=False)

        t0 = time.time()
        model = PatchTSTModel(**arch_params)
        history = train_model(
            model, train_loader, val_loader, device,
            epochs=train_params["epochs"],
            lr=train_params["lr"],
            patience=train_params["patience"],
            weight_decay=train_params["weight_decay"],
        )

        # Predict
        full_ds = DemandDataset(c_seqs, c_covs, targets=None)
        preds_log1p = predict_model(model, full_ds, device, batch_size=train_params["batch_size"])
        preds_raw = np.maximum(np.expm1(preds_log1p), 0)

        for j, idx in enumerate(idxs):
            all_results.append({**metadata_rows[idx], "basefcst_pref": float(preds_raw[j])})

        print(f"      Cluster {ci}/{len(clusters)} '{cluster_label}': "
              f"{len(idxs)} seqs, {len(history['train_loss'])} epochs ({time.time() - t0:.1f}s)")

    # Handle DFUs with no cluster -> zero prediction
    unknown_idxs = indices_by_cluster.get("__unknown__", [])
    if unknown_idxs:
        print(f"      {len(unknown_idxs)} sequences with no cluster -> zero prediction")
        for idx in unknown_idxs:
            all_results.append({**metadata_rows[idx], "basefcst_pref": 0.0})

    return all_results


def _train_and_predict_transfer(
    sequences: np.ndarray,
    covariates: np.ndarray,
    targets: np.ndarray,
    metadata_rows: list[dict],
    dfu_attrs_enc: pd.DataFrame,
    device,
    arch_params: dict,
    train_params: dict,
    transfer_epochs: int = 10,
    transfer_min_rows: int = 20,
    transfer_freeze_layers: int = 1,
) -> list[dict]:
    """Transfer learning: global base model -> per-cluster fine-tune with frozen layers."""
    import torch
    from torch.utils.data import DataLoader, random_split

    from scripts.patchtst_model import (
        DemandDataset,
        PatchTSTModel,
        freeze_for_transfer,
        predict_model,
        train_model,
        unfreeze_all,
    )

    n_total = len(sequences)
    if n_total == 0:
        return []

    # ── Phase 1: Train global base model ─────────────────────────────────
    print(f"      Phase 1: Training global base PatchTST ({n_total:,} sequences)...")
    t0 = time.time()

    dataset = DemandDataset(sequences, covariates, targets)
    n_val = max(int(n_total * 0.2), 1)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=train_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=train_params["batch_size"], shuffle=False)

    base_model = PatchTSTModel(**arch_params)
    history = train_model(
        base_model, train_loader, val_loader, device,
        epochs=train_params["epochs"],
        lr=train_params["lr"],
        patience=train_params["patience"],
        weight_decay=train_params["weight_decay"],
    )
    print(f"      Base model trained: {len(history['train_loss'])} epochs, "
          f"best_val_loss={min(history['val_loss']):.4f} ({time.time() - t0:.1f}s)")

    # Save base model state for cloning
    base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    # ── Phase 2: Per-cluster fine-tune ───────────────────────────────────
    dfu_cluster = dfu_attrs_enc.set_index("dfu_ck")["ml_cluster"].to_dict()
    row_clusters = [dfu_cluster.get(m["dfu_ck"], "__unknown__") for m in metadata_rows]
    row_clusters = [c if c is not None and str(c) != "nan" else "__unknown__" for c in row_clusters]

    clusters = sorted(set(c for c in row_clusters if c != "__unknown__"))
    indices_by_cluster: dict[str, list[int]] = {}
    for i, c in enumerate(row_clusters):
        indices_by_cluster.setdefault(c, []).append(i)

    all_results: list[dict] = [None] * n_total  # type: ignore[list-item]
    fine_tuned_clusters = 0

    print(f"      Phase 2: Fine-tuning {len(clusters)} clusters "
          f"(min_rows={transfer_min_rows}, freeze_layers={transfer_freeze_layers}, "
          f"transfer_epochs={transfer_epochs})...")

    for ci, cluster_label in enumerate(clusters, 1):
        idxs = indices_by_cluster.get(cluster_label, [])

        if len(idxs) == 0:
            continue

        if len(idxs) < transfer_min_rows:
            # Fallback: use base model predictions
            print(f"      Cluster {ci}/{len(clusters)} '{cluster_label}': "
                  f"{len(idxs)} < {transfer_min_rows} -> base model fallback")
            c_seqs = sequences[idxs]
            c_covs = covariates[idxs]
            pred_ds = DemandDataset(c_seqs, c_covs, targets=None)
            base_model.load_state_dict({k: v.to(device) for k, v in base_state.items()})
            preds_log1p = predict_model(base_model, pred_ds, device, batch_size=train_params["batch_size"])
            preds_raw = np.maximum(np.expm1(preds_log1p), 0)
            for j, idx in enumerate(idxs):
                all_results[idx] = {**metadata_rows[idx], "basefcst_pref": float(preds_raw[j])}
            continue

        # Clone base model and fine-tune
        t1 = time.time()
        ft_model = PatchTSTModel(**arch_params)
        ft_model.load_state_dict(base_state)
        freeze_for_transfer(ft_model, freeze_layers=transfer_freeze_layers)

        c_seqs = sequences[idxs]
        c_covs = covariates[idxs]
        c_tgts = targets[idxs]

        c_dataset = DemandDataset(c_seqs, c_covs, c_tgts)
        c_n_val = max(int(len(idxs) * 0.2), 1)
        c_n_train = len(idxs) - c_n_val
        c_train_ds, c_val_ds = random_split(c_dataset, [c_n_train, c_n_val])

        c_train_loader = DataLoader(c_train_ds, batch_size=train_params["batch_size"], shuffle=True)
        c_val_loader = DataLoader(c_val_ds, batch_size=train_params["batch_size"], shuffle=False)

        ft_history = train_model(
            ft_model, c_train_loader, c_val_loader, device,
            epochs=transfer_epochs,
            lr=train_params["lr"] * 0.5,  # Lower LR for fine-tuning
            patience=max(train_params["patience"] // 2, 2),
            weight_decay=train_params["weight_decay"],
        )

        # Predict with fine-tuned model
        pred_ds = DemandDataset(c_seqs, c_covs, targets=None)
        preds_log1p = predict_model(ft_model, pred_ds, device, batch_size=train_params["batch_size"])
        preds_raw = np.maximum(np.expm1(preds_log1p), 0)

        for j, idx in enumerate(idxs):
            all_results[idx] = {**metadata_rows[idx], "basefcst_pref": float(preds_raw[j])}

        fine_tuned_clusters += 1
        print(f"      Cluster {ci}/{len(clusters)} '{cluster_label}': "
              f"{len(idxs)} seqs, {len(ft_history['train_loss'])} epochs ({time.time() - t1:.1f}s)")

    # Handle DFUs with no cluster -> base model fallback
    unknown_idxs = indices_by_cluster.get("__unknown__", [])
    if unknown_idxs:
        print(f"      {len(unknown_idxs)} sequences with no cluster -> base model fallback")
        c_seqs = sequences[unknown_idxs]
        c_covs = covariates[unknown_idxs]
        pred_ds = DemandDataset(c_seqs, c_covs, targets=None)
        base_model.load_state_dict({k: v.to(device) for k, v in base_state.items()})
        preds_log1p = predict_model(base_model, pred_ds, device, batch_size=train_params["batch_size"])
        preds_raw = np.maximum(np.expm1(preds_log1p), 0)
        for j, idx in enumerate(unknown_idxs):
            all_results[idx] = {**metadata_rows[idx], "basefcst_pref": float(preds_raw[j])}

    # Fill any remaining None entries (shouldn't happen, but safety net)
    final_results = []
    for i, r in enumerate(all_results):
        if r is None:
            final_results.append({**metadata_rows[i], "basefcst_pref": 0.0})
        else:
            final_results.append(r)

    return final_results


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PatchTST backtest with expanding-window timeframes")
    parser.add_argument("--cluster-strategy", choices=["global", "per_cluster", "transfer"], default="global",
                        help="global: one model, per_cluster: model per ml_cluster, transfer: global base -> per-cluster fine-tune")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Override model_id (default: patchtst_global, patchtst_cluster, or patchtst_transfer)")
    parser.add_argument("--n-timeframes", type=int, default=10, help="Number of expanding windows")
    parser.add_argument("--output-dir", type=str, default="data/backtest", help="Output directory")

    # PatchTST architecture args
    parser.add_argument("--seq-len", type=int, default=12, help="Input sequence length (months)")
    parser.add_argument("--patch-len", type=int, default=3, help="Patch length (months)")
    parser.add_argument("--stride", type=int, default=2, help="Stride between patches")
    parser.add_argument("--d-model", type=int, default=64, help="Transformer embedding dimension")
    parser.add_argument("--nhead", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of Transformer encoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")

    # Training args
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=512, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")

    # Transfer learning args
    parser.add_argument("--transfer-epochs", type=int, default=10, help="Fine-tuning epochs for transfer strategy")
    parser.add_argument("--transfer-min-rows", type=int, default=20,
                        help="Minimum sequences per cluster for fine-tuning; smaller clusters use base model")
    parser.add_argument("--transfer-freeze-layers", type=int, default=1,
                        help="Number of Transformer encoder layers to freeze during fine-tuning")

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device override (auto: MPS > CUDA > CPU)")

    args = parser.parse_args()

    t_start = time.time()
    load_dotenv(ROOT / ".env")

    # Lazy import torch and model (so --help works without torch installed)
    import torch

    from scripts.patchtst_model import get_device

    device = get_device(args.device)

    _default_model_ids = {
        "global": "patchtst_global",
        "per_cluster": "patchtst_cluster",
        "transfer": "patchtst_transfer",
    }
    model_id = args.model_id or _default_model_ids[args.cluster_strategy]

    arch_params = {
        "seq_len": args.seq_len,
        "patch_len": args.patch_len,
        "stride": args.stride,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "n_covariates": 11,
    }

    train_params = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.learning_rate,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
    }

    print(f"[{_ts()}] PatchTST Backtest: strategy={args.cluster_strategy}, model_id={model_id}, "
          f"n_timeframes={args.n_timeframes}, device={device}")
    print(f"[{_ts()}] Architecture: {arch_params}")
    print(f"[{_ts()}] Training: {train_params}")

    # ── Step 1: Load data ────────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 1: Loading data from Postgres...")
    db = get_db_params()
    sales_df, dfu_attrs, item_attrs = load_backtest_data(db, include_item_attrs=True)

    exec_lag_map = dfu_attrs.set_index("dfu_ck")["execution_lag"].fillna(0).astype(int).to_dict()

    # ── Step 2: Encode categoricals ──────────────────────────────────────────
    print(f"\n[{_ts()}] Step 2: Encoding categorical features...")
    encoders, dfu_attrs_enc = _encode_categoricals(dfu_attrs)
    for col, le in encoders.items():
        print(f"  {col}: {len(le.classes_)} classes")

    # ── Step 3: Generate timeframes ──────────────────────────────────────────
    latest_month = sales_df["startdate"].max()
    earliest_month = sales_df["startdate"].min()
    print(f"\n[{_ts()}] Step 3: Date range: {earliest_month.date()} -> {latest_month.date()}")

    timeframes = generate_timeframes(earliest_month, latest_month, args.n_timeframes)
    print(f"  Generated {len(timeframes)} timeframes:")
    for tf in timeframes:
        print(f"  {tf['label']}: train [{tf['train_start'].date()} -> {tf['train_end'].date()}], "
              f"predict [{tf['predict_start'].date()} -> {tf['predict_end'].date()}]")

    all_months = sorted(sales_df["startdate"].unique())

    # ── Step 4: Build sales pivot table ──────────────────────────────────────
    print(f"\n[{_ts()}] Step 4: Building sales pivot table...")
    sales_pivot = _build_sales_pivot(sales_df, all_months)
    print(f"  Pivot shape: {sales_pivot.shape[0]:,} DFUs x {sales_pivot.shape[1]} months")

    # ── Step 5: Train & predict per timeframe ────────────────────────────────
    print(f"\n[{_ts()}] Step 5: Running {len(timeframes)} timeframe backtests...")
    all_predictions = []

    for ti, tf in enumerate(timeframes):
        label = tf["label"]
        train_end = tf["train_end"]
        predict_start = tf["predict_start"]
        predict_end = tf["predict_end"]
        tf_start = time.time()

        print(f"\n-- Timeframe {label} ({ti + 1}/{len(timeframes)}) --")

        predict_months = sorted([m for m in all_months if predict_start <= m <= predict_end])
        if not predict_months:
            print(f"  [{_ts()}] No predict months -- skipping")
            continue

        train_months = [m for m in all_months if earliest_month <= m <= train_end]
        if len(train_months) < 3:
            print(f"  [{_ts()}] Insufficient training months ({len(train_months)}) -- need 3 min -- skipping")
            continue

        # Build sequences for this timeframe
        print(f"  [{_ts()}] Building sequences ({len(predict_months)} predict months)...")
        t1 = time.time()
        sequences, covariates_arr, targets, metadata_rows = _build_sequences_for_timeframe(
            sales_pivot, dfu_attrs_enc, item_attrs,
            predict_months, all_months, train_end,
            seq_len=args.seq_len,
        )
        print(f"  [{_ts()}] Built {len(sequences):,} sequences ({time.time() - t1:.1f}s)")

        if len(sequences) == 0:
            print(f"  [{_ts()}] No sequences -- skipping")
            continue

        # Train & predict based on strategy
        print(f"  [{_ts()}] Strategy: {args.cluster_strategy}")
        if args.cluster_strategy == "global":
            pred_rows = _train_and_predict_global(
                sequences, covariates_arr, targets, metadata_rows,
                device, arch_params, train_params,
            )
        elif args.cluster_strategy == "per_cluster":
            pred_rows = _train_and_predict_per_cluster(
                sequences, covariates_arr, targets, metadata_rows,
                dfu_attrs_enc, device, arch_params, train_params,
            )
        elif args.cluster_strategy == "transfer":
            pred_rows = _train_and_predict_transfer(
                sequences, covariates_arr, targets, metadata_rows,
                dfu_attrs_enc, device, arch_params, train_params,
                transfer_epochs=args.transfer_epochs,
                transfer_min_rows=args.transfer_min_rows,
                transfer_freeze_layers=args.transfer_freeze_layers,
            )
        else:
            raise ValueError(f"Unknown cluster strategy: {args.cluster_strategy}")

        if not pred_rows:
            print(f"  [{_ts()}] No predictions -- skipping")
            continue

        preds_df = pd.DataFrame(pred_rows)
        preds_df["model_id"] = model_id
        preds_df["timeframe"] = label
        preds_df["timeframe_idx"] = tf["index"]
        all_predictions.append(preds_df)
        print(f"  [{_ts()}] Timeframe {label} complete: {len(preds_df):,} predictions ({time.time() - tf_start:.1f}s)")

        # Clear GPU cache between timeframes
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_predictions:
        print(f"\n[{_ts()}] No predictions generated. Check data range and timeframe count.")
        sys.exit(1)

    # ── Step 6: Combine, assign execution lag, attach actuals ────────────────
    print(f"\n[{_ts()}] Step 6: Combining predictions...")
    expanded, archive_expanded, combined = postprocess_predictions(
        all_predictions, sales_df, exec_lag_map
    )

    # ── Step 7: Save output ──────────────────────────────────────────────────
    print(f"\n[{_ts()}] Step 7: Saving output...")
    output_dir = ROOT / args.output_dir
    patchtst_params = {**arch_params, **train_params}
    if args.cluster_strategy == "transfer":
        patchtst_params["transfer_epochs"] = args.transfer_epochs
        patchtst_params["transfer_min_rows"] = args.transfer_min_rows
        patchtst_params["transfer_freeze_layers"] = args.transfer_freeze_layers

    output_path, archive_path, meta_path, metadata = save_backtest_output(
        output_df=expanded,
        archive_df=archive_expanded,
        output_dir=output_dir,
        model_id=model_id,
        cluster_strategy=args.cluster_strategy,
        n_timeframes=args.n_timeframes,
        model_params=patchtst_params,
        model_params_key="patchtst_params",
        timeframes=timeframes,
        earliest_month=earliest_month,
        latest_month=latest_month,
        extra_metadata={"device": str(device)},
    )

    # ── Step 8: MLflow logging ───────────────────────────────────────────────
    mlflow_hyperparams = {
        "n_timeframes": args.n_timeframes,
        "cluster_strategy": args.cluster_strategy,
        "seq_len": args.seq_len,
        "patch_len": args.patch_len,
        "stride": args.stride,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
    }
    if args.cluster_strategy == "transfer":
        mlflow_hyperparams["transfer_epochs"] = args.transfer_epochs
        mlflow_hyperparams["transfer_min_rows"] = args.transfer_min_rows
        mlflow_hyperparams["transfer_freeze_layers"] = args.transfer_freeze_layers

    log_backtest_run(
        model_type="patchtst_backtest",
        model_id=model_id,
        cluster_strategy=args.cluster_strategy,
        hyperparams=mlflow_hyperparams,
        metrics={
            "n_predictions": len(expanded),
            "n_dfus": int(expanded["dmdunit"].nunique()),
        },
        metadata=metadata,
        artifact_paths=[str(output_path), str(archive_path), str(meta_path)],
    )

    elapsed = time.time() - t_start
    print(f"\n[{_ts()}] PatchTST backtest complete in {elapsed:.0f}s ({elapsed / 60:.1f}m)")


if __name__ == "__main__":
    main()
