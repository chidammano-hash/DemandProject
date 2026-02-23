"""PatchTST (Patch Time Series Transformer) model for demand forecasting.

Implements a lightweight PatchTST architecture tailored for monthly demand
prediction.  The model segments a 12-month quantity history into overlapping
3-month patches, projects them to a latent space, conditions on static
covariates, and passes through a Transformer encoder to produce a single
next-month forecast.

Architecture
------------
1. Patch embedding   : Linear(patch_len, d_model) over overlapping patches
2. Positional enc.   : Learnable positional encoding (n_patches)
3. Covariate inject. : Linear(n_covariates, d_model) added to first patch token
4. Transformer enc.  : num_layers x (MultiHeadAttention + FFN, GELU activation)
5. MLP head          : Flatten -> Linear -> single scalar output

All quantities are in log1p space during training; callers expm1 the output.

References
----------
Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with
Transformers", ICLR 2023.
"""

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ── Device selection ─────────────────────────────────────────────────────────


def get_device(override: Optional[str] = None) -> torch.device:
    """Auto-detect best available device: MPS > CUDA > CPU.

    Args:
        override: Force a specific device string (e.g. "cpu", "cuda", "mps").

    Returns:
        torch.device for the selected backend.
    """
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Dataset ──────────────────────────────────────────────────────────────────


class DemandDataset(Dataset):
    """PyTorch dataset for (sequence, covariates, target) triplets.

    Args:
        sequences:  np.ndarray of shape (N, seq_len) — log1p qty values.
        covariates: np.ndarray of shape (N, n_covariates) — static + calendar.
        targets:    np.ndarray of shape (N,) — log1p qty of target month.
                    Pass None for inference-only datasets (targets = zeros).
    """

    def __init__(
        self,
        sequences: np.ndarray,
        covariates: np.ndarray,
        targets: Optional[np.ndarray] = None,
    ) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.covariates = torch.tensor(covariates, dtype=torch.float32)
        if targets is not None:
            self.targets = torch.tensor(targets, dtype=torch.float32)
        else:
            self.targets = torch.zeros(len(sequences), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.covariates[idx], self.targets[idx]


# ── PatchTST Model ───────────────────────────────────────────────────────────


class PatchTSTModel(nn.Module):
    """Patch Time Series Transformer for single-step demand forecasting.

    Parameters
    ----------
    seq_len : int
        Input sequence length (default 12 months).
    patch_len : int
        Length of each patch (default 3 months).
    stride : int
        Stride between consecutive patches (default 2).
    d_model : int
        Transformer embedding dimension (default 64).
    nhead : int
        Number of attention heads (default 4).
    num_layers : int
        Number of Transformer encoder layers (default 2).
    dropout : float
        Dropout rate (default 0.1).
    n_covariates : int
        Number of static covariate features (default 11).

    Forward pass
    ------------
    1. Reshape input (batch, seq_len) into patches (batch, n_patches, patch_len).
    2. Project patches via Linear(patch_len, d_model).
    3. Add learnable positional encoding.
    4. Project covariates to d_model, inject into first patch token (addition).
    5. Pass through TransformerEncoder (num_layers, nhead, GELU).
    6. Flatten -> Linear -> single scalar output.
    """

    def __init__(
        self,
        seq_len: int = 12,
        patch_len: int = 3,
        stride: int = 2,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_covariates: int = 11,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model

        # Number of patches: (seq_len - patch_len) // stride + 1
        self.n_patches = (seq_len - patch_len) // stride + 1

        # Patch projection: (batch, n_patches, patch_len) -> (batch, n_patches, d_model)
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Learnable positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Covariate projection: inject into first patch token
        self.cov_proj = nn.Linear(n_covariates, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP head: flatten all patch embeddings -> single output
        self.head = nn.Sequential(
            nn.Linear(self.n_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, seq: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            seq: (batch, seq_len) — log1p qty input sequence.
            cov: (batch, n_covariates) — static + calendar covariates.

        Returns:
            (batch,) — predicted log1p qty for next month.
        """
        batch_size = seq.size(0)

        # 1. Create patches: unfold along the time dimension
        # (batch, seq_len) -> (batch, n_patches, patch_len)
        patches = seq.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # 2. Project patches to d_model
        # (batch, n_patches, patch_len) -> (batch, n_patches, d_model)
        x = self.patch_proj(patches)

        # 3. Add positional encoding
        x = x + self.pos_encoding

        # 4. Covariate conditioning: project and add to first patch token
        cov_embed = self.cov_proj(cov)  # (batch, d_model)
        x[:, 0, :] = x[:, 0, :] + cov_embed

        # 5. Transformer encoder
        x = self.transformer(x)  # (batch, n_patches, d_model)

        # 6. Flatten and MLP head
        x = x.reshape(batch_size, -1)  # (batch, n_patches * d_model)
        out = self.head(x).squeeze(-1)  # (batch,)

        return out


# ── Training ─────────────────────────────────────────────────────────────────


def train_model(
    model: PatchTSTModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    weight_decay: float = 1e-4,
) -> dict:
    """Train PatchTST with HuberLoss + AdamW + CosineAnnealingLR + early stopping.

    Args:
        model:        PatchTSTModel instance (already on device).
        train_loader: Training DataLoader.
        val_loader:   Validation DataLoader (None to skip early stopping).
        device:       torch.device.
        epochs:       Maximum training epochs.
        lr:           Learning rate for AdamW.
        patience:     Early stopping patience (epochs without val loss improvement).
        weight_decay: L2 regularization weight.

    Returns:
        Dict with training history: {"train_loss": [...], "val_loss": [...], "best_epoch": int}.
    """
    model.to(device)
    model.train()

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    for epoch in range(epochs):
        # ── Training phase ───────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        n_batches = 0

        for seq_batch, cov_batch, target_batch in train_loader:
            seq_batch = seq_batch.to(device)
            cov_batch = cov_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            output = model(seq_batch, cov_batch)
            loss = criterion(output, target_batch)
            loss.backward()

            # Gradient norm clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        scheduler.step()

        # ── Validation phase ─────────────────────────────────────────────
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for seq_batch, cov_batch, target_batch in val_loader:
                    seq_batch = seq_batch.to(device)
                    cov_batch = cov_batch.to(device)
                    target_batch = target_batch.to(device)

                    output = model(seq_batch, cov_batch)
                    loss = criterion(output, target_batch)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            history["val_loss"].append(avg_val_loss)

            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
                history["best_epoch"] = epoch
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break
        else:
            history["val_loss"].append(avg_train_loss)

    # Restore best model state
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return history


# ── Prediction ───────────────────────────────────────────────────────────────


def predict_model(
    model: PatchTSTModel,
    dataset: DemandDataset,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run inference and return predictions in log1p space.

    Args:
        model:      Trained PatchTSTModel.
        dataset:    DemandDataset (targets are ignored).
        device:     torch.device.
        batch_size: Inference batch size.

    Returns:
        np.ndarray of shape (N,) — predicted values in log1p space.
    """
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds = []

    with torch.no_grad():
        for seq_batch, cov_batch, _ in loader:
            seq_batch = seq_batch.to(device)
            cov_batch = cov_batch.to(device)
            output = model(seq_batch, cov_batch)
            all_preds.append(output.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


# ── Transfer learning helpers ────────────────────────────────────────────────


def freeze_for_transfer(model: PatchTSTModel, freeze_layers: int = 1) -> None:
    """Freeze patch embedding + first N Transformer encoder layers for transfer learning.

    This preserves the learned patch representations and low-level temporal
    patterns from the global model while allowing the higher layers and MLP
    head to adapt to cluster-specific demand patterns.

    Args:
        model:         PatchTSTModel instance.
        freeze_layers: Number of Transformer encoder layers to freeze (from bottom).
    """
    # Freeze patch projection
    for param in model.patch_proj.parameters():
        param.requires_grad = False

    # Freeze positional encoding
    model.pos_encoding.requires_grad = False

    # Freeze first N encoder layers
    for i, layer in enumerate(model.transformer.layers):
        if i < freeze_layers:
            for param in layer.parameters():
                param.requires_grad = False


def unfreeze_all(model: PatchTSTModel) -> None:
    """Unfreeze all parameters (restore full trainability).

    Args:
        model: PatchTSTModel instance.
    """
    for param in model.parameters():
        param.requires_grad = True
    model.pos_encoding.requires_grad = True
