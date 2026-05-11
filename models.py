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
