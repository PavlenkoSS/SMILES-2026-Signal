import math

import torch
import torch.nn as nn


class MLPInterferenceCanceller(nn.Module):
    """Simple MLP baseline: flattens window, predicts full window output."""

    def __init__(self, window_size=512, in_features=12, out_features=8, hidden=512):
        super().__init__()
        self.window_size = window_size
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(window_size * in_features, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, window_size * out_features),
        )

    def forward(self, x):
        # x: (B, T, 12)
        B = x.shape[0]
        out = self.net(x.reshape(B, -1))
        return out.reshape(B, self.window_size, self.out_features)


class ResidualBlock1D(nn.Module):
    """1D residual block with same-padding convolutions."""

    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class CNNGRUCanceller(nn.Module):
    """CNN frontend + GRU backend for temporal interference prediction."""

    def __init__(self, in_ch=12, out_ch=8, cnn_ch=64, gru_hidden=128, gru_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, cnn_ch, kernel_size=7, padding=3),
            nn.GELU(),
            nn.GroupNorm(8, cnn_ch),
            ResidualBlock1D(cnn_ch, cnn_ch, kernel_size=5),
            ResidualBlock1D(cnn_ch, cnn_ch * 2, kernel_size=5),
            ResidualBlock1D(cnn_ch * 2, cnn_ch * 2, kernel_size=5, dilation=2),
            ResidualBlock1D(cnn_ch * 2, cnn_ch * 2, kernel_size=5, dilation=4),
        )
        self.gru = nn.GRU(
            input_size=cnn_ch * 2,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(gru_hidden, 64),
            nn.GELU(),
            nn.Linear(64, out_ch),
        )

    def forward(self, x):
        # x: (B, T, 12)
        h = x.transpose(1, 2)  # (B, 12, T)
        h = self.cnn(h)        # (B, 128, T)
        h = h.transpose(1, 2)  # (B, T, 128)
        h, _ = self.gru(h)     # (B, T, gru_hidden)
        return self.head(h)    # (B, T, 8)


def sinusoidal_pe(length: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Standard sinusoidal positional encoding, shape (1, length, d_model)."""
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(length, d_model, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


class TransformerInterferenceCanceller(nn.Module):
    """Encoder-only Transformer: TX time series -> per-timestep interference (8 real channels).

    Uses full (non-causal) self-attention over the window, suitable for offline batch scoring.
    """

    def __init__(
        self,
        in_ch: int = 12,
        out_ch: int = 8,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        self.d_model = d_model
        self.input_proj = nn.Linear(in_ch, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 12)
        B, T, _ = x.shape
        h = self.input_proj(x)
        h = h + sinusoidal_pe(T, self.d_model, h.device, h.dtype)
        h = self.encoder(h)
        return self.head(h)
