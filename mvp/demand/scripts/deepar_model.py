"""DeepAR (Deep AutoRegressive) model for demand forecasting.

Implements a DeepAR architecture tailored for monthly demand prediction.
The model uses a stacked LSTM encoder conditioned on static covariates,
with a Gaussian distribution head that outputs (mu, sigma) parameters
for probabilistic forecasting.  The point forecast is simply mu.

Architecture
------------
1. Covariate projection : Linear(n_covariates, hidden_size)
2. LSTM encoder         : Stacked LSTM over (qty_t, cov_proj) at each timestep
3. Gaussian head        : hidden -> mu (Linear) + sigma (Linear + Softplus)

All quantities are in log1p space during training; callers expm1 the output.

References
----------
Salinas et al., "DeepAR: Probabilistic Forecasting with Autoregressive
Recurrent Networks", International Journal of Forecasting, 2020.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# -- Device selection ---------------------------------------------------------


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


# -- Dataset ------------------------------------------------------------------


class DemandDataset(Dataset):
    """PyTorch dataset for (sequence, covariates, target) triplets.

    Args:
        sequences:  np.ndarray of shape (N, seq_len) -- log1p qty values.
        covariates: np.ndarray of shape (N, n_covariates) -- static + calendar.
        targets:    np.ndarray of shape (N,) -- log1p qty of target month.
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


# -- DeepAR Model -------------------------------------------------------------


class DeepARModel(nn.Module):
    """DeepAR autoregressive LSTM for single-step demand forecasting.

    Parameters
    ----------
    seq_len : int
        Input sequence length (default 12 months).
    hidden_size : int
        LSTM hidden dimension (default 64).
    num_layers : int
        Number of stacked LSTM layers (default 2).
    dropout : float
        Dropout rate between LSTM layers (default 0.1).
    n_covariates : int
        Number of static covariate features (default 11).

    Forward pass
    ------------
    1. Project covariates to hidden_size via Linear.
    2. Expand projected covariates to match seq_len (broadcast across time).
    3. Concatenate seq (unsqueeze -1) with expanded covariate projection.
    4. Feed through stacked LSTM.
    5. Take the last hidden state (output at final timestep).
    6. mu = mu_head(last_hidden).squeeze(-1)
    7. sigma = sigma_head(last_hidden).squeeze(-1)  (Softplus ensures > 0)
    8. Return (mu, sigma).
    """

    def __init__(
        self,
        seq_len: int = 12,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_covariates: int = 11,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Covariate projection: (batch, n_covariates) -> (batch, hidden_size)
        self.cov_proj = nn.Linear(n_covariates, hidden_size)

        # LSTM encoder: input_size = hidden_size (covariate proj) + 1 (qty value)
        self.lstm = nn.LSTM(
            input_size=hidden_size + 1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Gaussian distribution head
        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialization for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, seq: torch.Tensor, cov: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            seq: (batch, seq_len) -- log1p qty input sequence.
            cov: (batch, n_covariates) -- static + calendar covariates.

        Returns:
            (mu, sigma) each of shape (batch,) -- Gaussian parameters in log1p space.
        """
        batch_size = seq.size(0)

        # 1. Project covariates to hidden_size
        # (batch, n_covariates) -> (batch, hidden_size)
        cov_embed = self.cov_proj(cov)

        # 2. Expand covariates to match seq_len
        # (batch, hidden_size) -> (batch, seq_len, hidden_size)
        cov_expanded = cov_embed.unsqueeze(1).expand(-1, self.seq_len, -1)

        # 3. Concatenate seq (unsqueeze -1) with expanded covariates
        # (batch, seq_len, 1) cat (batch, seq_len, hidden_size) -> (batch, seq_len, hidden_size + 1)
        seq_input = seq.unsqueeze(-1)  # (batch, seq_len, 1)
        lstm_input = torch.cat([seq_input, cov_expanded], dim=-1)

        # 4. Feed through LSTM
        # output: (batch, seq_len, hidden_size)
        output, _ = self.lstm(lstm_input)

        # 5. Take last hidden state (output at final timestep)
        last_hidden = output[:, -1, :]  # (batch, hidden_size)

        # 6. mu from linear head
        mu = self.mu_head(last_hidden).squeeze(-1)  # (batch,)

        # 7. sigma from linear + softplus head
        sigma = self.sigma_head(last_hidden).squeeze(-1)  # (batch,)

        return mu, sigma


# -- Training -----------------------------------------------------------------


def train_model(
    model: DeepARModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    weight_decay: float = 1e-4,
) -> dict:
    """Train DeepAR with GaussianNLLLoss + AdamW + CosineAnnealingLR + early stopping.

    Args:
        model:        DeepARModel instance (already on device).
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

    criterion = nn.GaussianNLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

    for epoch in range(epochs):
        # -- Training phase ---------------------------------------------------
        model.train()
        total_loss = 0.0
        n_batches = 0

        for seq_batch, cov_batch, target_batch in train_loader:
            seq_batch = seq_batch.to(device)
            cov_batch = cov_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            mu, sigma = model(seq_batch, cov_batch)
            # GaussianNLLLoss expects (input=mu, target, var=sigma**2)
            loss = criterion(mu, target_batch, sigma ** 2)
            loss.backward()

            # Gradient norm clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        scheduler.step()

        # -- Validation phase -------------------------------------------------
        if val_loader is not None and len(val_loader) > 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0

            with torch.no_grad():
                for seq_batch, cov_batch, target_batch in val_loader:
                    seq_batch = seq_batch.to(device)
                    cov_batch = cov_batch.to(device)
                    target_batch = target_batch.to(device)

                    mu, sigma = model(seq_batch, cov_batch)
                    loss = criterion(mu, target_batch, sigma ** 2)
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


# -- Prediction ---------------------------------------------------------------


def predict_model(
    model: DeepARModel,
    dataset: DemandDataset,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference and return (mu, sigma) arrays in log1p space.

    Args:
        model:      Trained DeepARModel.
        dataset:    DemandDataset (targets are ignored).
        device:     torch.device.
        batch_size: Inference batch size.

    Returns:
        (mu_array, sigma_array) each of shape (N,) -- values in log1p space.
    """
    model.to(device)
    model.eval()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_mu = []
    all_sigma = []

    with torch.no_grad():
        for seq_batch, cov_batch, _ in loader:
            seq_batch = seq_batch.to(device)
            cov_batch = cov_batch.to(device)
            mu, sigma = model(seq_batch, cov_batch)
            all_mu.append(mu.cpu().numpy())
            all_sigma.append(sigma.cpu().numpy())

    return np.concatenate(all_mu, axis=0), np.concatenate(all_sigma, axis=0)


# -- Transfer learning helpers ------------------------------------------------


def freeze_for_transfer(model: DeepARModel, freeze_layers: int = 1) -> None:
    """Freeze covariate projection + first N LSTM layers for transfer learning.

    This preserves the learned temporal patterns and covariate representations
    from the global model while allowing the higher LSTM layers and Gaussian
    head to adapt to cluster-specific demand patterns.

    Args:
        model:         DeepARModel instance.
        freeze_layers: Number of LSTM layers to freeze (from bottom).
    """
    # Freeze covariate projection
    for param in model.cov_proj.parameters():
        param.requires_grad = False

    # Freeze first N LSTM layers by freezing their weight/bias parameters
    # LSTM parameters are named: weight_ih_l{i}, weight_hh_l{i}, bias_ih_l{i}, bias_hh_l{i}
    for i in range(freeze_layers):
        for name, param in model.lstm.named_parameters():
            if f"_l{i}" in name:
                param.requires_grad = False


def unfreeze_all(model: DeepARModel) -> None:
    """Unfreeze all parameters (restore full trainability).

    Args:
        model: DeepARModel instance.
    """
    for param in model.parameters():
        param.requires_grad = True
