import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

from task_and_baseline import baseline, build_task_helpers


def complex_to_real(z):
    """(N, C) complex -> (N, 2C) float32: [real, imag] concatenated."""
    return np.concatenate([z.real, z.imag], axis=-1).astype(np.float32)


def real_to_complex(r):
    """(N, 2C) float32 -> (N, C) complex128: inverse of complex_to_real."""
    half = r.shape[-1] // 2
    return r[..., :half].astype(np.float64) + 1j * r[..., half:].astype(np.float64)


def load_data(mat_path="challenge.mat"):
    """Load challenge data, normalize TX, build helpers."""
    data = loadmat(mat_path, simplify_cells=True)
    tx = data["tx"].astype(np.complex128)
    rx = data["rx"].astype(np.complex128)
    Fs = float(data["Fs"])
    N, _ = tx.shape

    tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)
    helpers = build_task_helpers(tx_n, Fs, N)

    return tx_n, rx, Fs, N, helpers


def compute_baseline_interference(tx_n, rx, helpers):
    """Compute interference = rx - baseline(rx). Shape (N, 4) complex."""
    rx_baseline = baseline(tx_n, rx, helpers["fit_tx_prediction"])
    return rx - rx_baseline


def compute_enhanced_interference(tx_n, rx, helpers, lambda_rank1=1.0):
    """Compute enhanced interference: baseline TX pred + rank-1 component."""
    from enhanced_baseline import enhanced_canceller
    rx_hat = enhanced_canceller(tx_n, rx, helpers, lambda_rank1=lambda_rank1)
    return rx - rx_hat


class WindowedDataset(Dataset):
    """Sliding-window dataset with lazy indexing into numpy arrays."""

    def __init__(self, tx_real, target_real, window_size=512, stride=128):
        self.tx_real = tx_real
        self.target_real = target_real
        self.window_size = window_size
        N = len(tx_real)
        self.starts = list(range(0, N - window_size + 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.window_size
        x = torch.from_numpy(self.tx_real[s:e].copy())
        y = torch.from_numpy(self.target_real[s:e].copy())
        return x, y


def make_datasets(tx_n, rx, helpers, window_size=512, stride_train=128, stride_val=512,
                  train_frac=0.8, enhanced=False, lambda_rank1=1.0):
    """Create train/val WindowedDatasets with chronological split."""
    if enhanced:
        interference = compute_enhanced_interference(tx_n, rx, helpers, lambda_rank1)
    else:
        interference = compute_baseline_interference(tx_n, rx, helpers)
    tx_real = complex_to_real(tx_n)       # (N, 12)
    target_real = complex_to_real(interference)  # (N, 8)

    N = len(tx_real)
    split = int(N * train_frac)

    train_ds = WindowedDataset(tx_real[:split], target_real[:split], window_size, stride_train)
    val_ds = WindowedDataset(tx_real[split:], target_real[split:], window_size, stride_val)

    return train_ds, val_ds, tx_real, target_real
