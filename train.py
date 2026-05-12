"""Training script for neural SIC models."""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import load_data, make_datasets, complex_to_real
from device_utils import configure_cuda_for_training, pick_torch_device
from models import MLPInterferenceCanceller, CNNGRUCanceller, TransformerInterferenceCanceller


FS = 7_680_000.0
CENTER = 1.9e6
BW = 0.6e6


def band_power_loss(rx_hat, fs=FS, center=CENTER, bw=BW):
    """Compute mean power in target band using FFT. rx_hat: (B, T, 8)."""
    B, T, C = rx_hat.shape
    spec = torch.fft.fft(rx_hat, dim=1)  # (B, T, 8)
    freqs = torch.fft.fftfreq(T, d=1.0 / fs, device=rx_hat.device)
    mask = ((freqs >= center - bw / 2) & (freqs <= center + bw / 2)).float()  # (T,)
    mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, T, 1)
    band_power = (spec.abs() ** 2 * mask).mean()
    return band_power


def smoothness_loss(pred):
    """Temporal smoothness: MSE of consecutive differences."""
    diff = pred[:, 1:] - pred[:, :-1]
    return (diff ** 2).mean()


def train_epoch(model, loader, optimizer, rx_real_all, alpha, beta, gamma, device):
    model.train()
    total_loss = 0.0
    n = 0
    for tx_win, target_win in loader:
        nb = device.type == "cuda"
        tx_win = tx_win.to(device, non_blocking=nb)
        target_win = target_win.to(device, non_blocking=nb)

        pred = model(tx_win)
        mse = nn.functional.mse_loss(pred, target_win)

        loss = alpha * mse
        if beta > 0:
            loss = loss + beta * band_power_loss(target_win - pred + tx_win[:, :, :8])
        if gamma > 0:
            loss = loss + gamma * smoothness_loss(pred)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * tx_win.shape[0]
        n += tx_win.shape[0]

    return total_loss / max(n, 1)


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    total_mse = 0.0
    n = 0
    for tx_win, target_win in loader:
        nb = device.type == "cuda"
        tx_win = tx_win.to(device, non_blocking=nb)
        target_win = target_win.to(device, non_blocking=nb)
        pred = model(tx_win)
        mse = nn.functional.mse_loss(pred, target_win)
        total_mse += mse.item() * tx_win.shape[0]
        n += tx_win.shape[0]
    return total_mse / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "cnngru", "transformer"], default="cnngru")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--window_size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1e-4)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers; >0 can help pipeline data on Linux+CUDA (default 0)",
    )
    parser.add_argument("--enhanced", action="store_true",
                        help="Use enhanced target (baseline + rank1)")
    args = parser.parse_args()

    device = pick_torch_device(args.device if args.device.strip() else None)
    configure_cuda_for_training(device)
    print(f"Using device: {device}")

    print("Loading data...")
    tx_n, rx, Fs, N, helpers = load_data()

    print("Building datasets...")
    train_ds, val_ds, tx_real, target_real = make_datasets(
        tx_n, rx, helpers,
        window_size=args.window_size,
        stride_train=args.stride,
        stride_val=args.window_size,
        enhanced=args.enhanced,
    )
    print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    if args.model == "mlp":
        model = MLPInterferenceCanceller(window_size=args.window_size)
    elif args.model == "cnngru":
        model = CNNGRUCanceller()
    else:
        model = TransformerInterferenceCanceller()
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}, params: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, tx_real, args.alpha, args.beta, args.gamma, device
        )
        val_loss = val_epoch(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        improved = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.checkpoint)
            improved = " *"

        print(f"Epoch {epoch:3d}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s{improved}")

    print(f"\nBest val loss: {best_val:.6f}")
    print(f"Checkpoint saved: {args.checkpoint}")


if __name__ == "__main__":
    main()
