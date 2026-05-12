"""Phi-only neural cancellers (TX-derived features, no raw RX in forward)."""
import torch
import torch.nn as nn


class FeatureMLPCanceller(nn.Module):
    """Per-sample MLP: (B, T, in_dim) -> (B, T, 8) real/imag of 4 complex channels."""

    def __init__(self, in_dim: int, hidden: int = 512):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        if d != self.in_dim:
            raise ValueError(f"expected in_dim {self.in_dim}, got {d}")
        y = self.net(x.reshape(b * t, d))
        return y.reshape(b, t, 8)


class FeatureCNNCanceller(nn.Module):
    """1D CNN over time with in_ch = 2*K (real concat imag per lag-feature column)."""

    def __init__(self, in_dim: int, ch: int = 128):
        super().__init__()
        self.in_dim = in_dim
        self.front = nn.Sequential(
            nn.Conv1d(in_dim, ch, kernel_size=7, padding=3),
            nn.GELU(),
            nn.GroupNorm(8, ch),
            nn.Conv1d(ch, ch, kernel_size=5, padding=2),
            nn.GELU(),
            nn.GroupNorm(8, ch),
        )
        self.head = nn.Conv1d(ch, 8, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim) -> (B, in_dim, T)
        x = x.transpose(1, 2)
        h = self.front(x)
        return self.head(h).transpose(1, 2)
