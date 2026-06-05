from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class GRUAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.encoder = nn.GRU(n_features, hidden_size, batch_first=True)
        self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.encoder(inputs)
        repeated = hidden[-1].unsqueeze(1).repeat(1, inputs.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.activation = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(inputs)[..., : inputs.shape[-1]]) + inputs


class TCNAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.input = nn.Conv1d(n_features, hidden_size, kernel_size=1)
        self.blocks = nn.Sequential(TCNBlock(hidden_size, 1), TCNBlock(hidden_size, 2), TCNBlock(hidden_size, 4))
        self.output = nn.Conv1d(hidden_size, n_features, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        channels_first = inputs.transpose(1, 2)
        encoded = self.blocks(self.input(channels_first))
        return self.output(encoded).transpose(1, 2)


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, n_heads: int = 4):
        super().__init__()
        self.input = nn.Linear(n_features, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 2,
            batch_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output = nn.Linear(hidden_size, n_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.encoder(self.input(inputs)))


class TSMixerBlock(nn.Module):
    def __init__(self, window_size: int, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.token_mixer = nn.Sequential(nn.Linear(window_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, window_size))
        self.channel_mixer = nn.Sequential(nn.Linear(n_features, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_features))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        mixed_tokens = self.token_mixer(inputs.transpose(1, 2)).transpose(1, 2)
        inputs = inputs + mixed_tokens
        return inputs + self.channel_mixer(inputs)


class TSMixerAutoencoder(nn.Module):
    def __init__(self, n_features: int, window_size: int, hidden_size: int = 32):
        super().__init__()
        self.blocks = nn.Sequential(
            TSMixerBlock(window_size, n_features, hidden_size),
            TSMixerBlock(window_size, n_features, hidden_size),
        )
        self.output = nn.Linear(n_features, n_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output(self.blocks(inputs))


def build_autoencoder(model_name: str, *, n_features: int, window_size: int, hidden_size: int = 32) -> nn.Module:
    normalized = model_name.lower()
    if normalized == "gru":
        return GRUAutoencoder(n_features, hidden_size)
    if normalized == "tcn":
        return TCNAutoencoder(n_features, hidden_size)
    if normalized == "transformer":
        return TransformerAutoencoder(n_features, hidden_size)
    if normalized == "tsmixer":
        return TSMixerAutoencoder(n_features, window_size, hidden_size)
    raise ValueError(f"Unknown autoencoder model: {model_name}")


def train_autoencoder(
    model: nn.Module,
    train_windows: np.ndarray,
    validation_windows: np.ndarray,
    *,
    epochs: int = 8,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, float]:
    torch.manual_seed(seed)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    dataset = TensorDataset(torch.from_numpy(train_windows))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = 0.0
    for _ in range(epochs):
        losses = []
        for (batch,) in loader:
            batch = batch.to(torch_device)
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        final_loss = float(np.mean(losses)) if losses else 0.0

    validation_loss = (
        float(np.mean(reconstruction_errors(model, validation_windows, device=device))) if len(validation_windows) else 0.0
    )
    return {"train_loss": final_loss, "validation_reconstruction_error": validation_loss}


def reconstruction_errors(
    model: nn.Module,
    windows: np.ndarray,
    *,
    batch_size: int = 256,
    device: str = "cpu",
) -> np.ndarray:
    if len(windows) == 0:
        return np.empty((0,), dtype=np.float32)
    torch_device = torch.device(device)
    model.to(torch_device)
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[start : start + batch_size]).to(torch_device)
            reconstructed = model(batch)
            error = torch.mean((reconstructed - batch) ** 2, dim=(1, 2))
            errors.append(error.cpu().numpy())
    return np.concatenate(errors).astype(np.float32)


def save_model(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def isolation_forest_scores(train_windows: np.ndarray, all_windows: np.ndarray, *, seed: int = 42) -> np.ndarray:
    flattened_train = train_windows.reshape(train_windows.shape[0], -1)
    flattened_all = all_windows.reshape(all_windows.shape[0], -1)
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return _centroid_distance_scores(flattened_train, flattened_all)

    model = IsolationForest(n_estimators=100, contamination="auto", random_state=seed)
    model.fit(flattened_train)
    return (-model.score_samples(flattened_all)).astype(np.float32)


def _centroid_distance_scores(flattened_train: np.ndarray, flattened_all: np.ndarray) -> np.ndarray:
    centroid = flattened_train.mean(axis=0)
    return np.linalg.norm(flattened_all - centroid, axis=1).astype(np.float32)
