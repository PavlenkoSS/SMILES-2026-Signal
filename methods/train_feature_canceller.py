#!/usr/bin/env python3
"""Train Phi-only MLP/CNN to match ridge interference (distill + optional band loss on residual)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .data_utils import complex_to_real, load_data
from .device_utils import configure_cuda_for_training, pick_torch_device
from .models_feature import FeatureCNNCanceller, FeatureMLPCanceller
from .phi_features import phi_window_real
from .ridge_sic import fit_ridge_weights, predict_poly_from_weights
from .sic_filters import make_bp_callable
from . import feature_utils as fu


FS = 7_680_000.0
CENTER = 1.9e6
BW = 0.6e6


def band_power_on_residual(residual_ri: torch.Tensor) -> torch.Tensor:
    """residual_ri: (B, T, 8) float — band-limited mean power in scorer band."""
    b, t, _c8 = residual_ri.shape
    spec = torch.fft.fft(residual_ri, dim=1)
    freqs = torch.fft.fftfreq(t, d=1.0 / FS, device=residual_ri.device)
    mask = ((freqs >= CENTER - BW / 2) & (freqs <= CENTER + BW / 2)).float()
    mask = mask.view(1, t, 1)
    return (spec.abs() ** 2 * mask).mean()


class PhiWindowDataset(Dataset):
    def __init__(
        self,
        tx_n,
        rx_res,
        I_teacher,
        bp,
        lags,
        orders_mem,
        orders_conj,
        use_conjugate,
        use_cross,
        cross_kinds,
        cross_lag_deltas,
        col_indices,
        starts,
        window_size,
    ):
        self.tx_n = tx_n
        self.rx_res = rx_res
        self.I_teacher = I_teacher
        self.bp = bp
        self.lags = lags
        self.orders_mem = orders_mem
        self.orders_conj = orders_conj
        self.use_conjugate = use_conjugate
        self.use_cross = use_cross
        self.cross_kinds = cross_kinds
        self.cross_lag_deltas = cross_lag_deltas
        self.col_indices = col_indices
        self.starts = starts
        self.window_size = window_size

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[idx])
        w = self.window_size
        phi = phi_window_real(
            self.tx_n,
            self.bp,
            self.lags,
            self.orders_mem,
            self.orders_conj,
            self.use_conjugate,
            self.use_cross,
            self.cross_kinds,
            self.cross_lag_deltas,
            self.col_indices,
            s,
            w,
        )
        tgt = complex_to_real(self.I_teacher[s : s + w])
        rx_ri = complex_to_real(self.rx_res[s : s + w])
        return (
            torch.from_numpy(phi),
            torch.from_numpy(tgt),
            torch.from_numpy(rx_ri),
        )


def _teacher_and_columns(tx_n, rx, helpers, Fs, lags, args, bp):
    fit_tx = helpers["fit_tx_prediction"]
    tx_pred = fit_tx(rx)
    rx_res = rx - tx_pred
    W, _, _ = fit_ridge_weights(
        tx_n,
        rx_res,
        helpers,
        lags,
        Fs,
        ridge_lambda=args.ridge_lambda,
        orders_mem=args.orders_mem,
        orders_conj=args.orders_conj,
        use_conjugate=args.use_conjugate,
        use_cross=args.use_cross,
        cross_kinds=args.cross_kinds,
        cross_lag_deltas=args.cross_lag_deltas,
    )
    I_t = predict_poly_from_weights(
        tx_n,
        bp,
        lags,
        W,
        args.orders_mem,
        args.orders_conj,
        args.use_conjugate,
        args.use_cross,
        args.cross_kinds,
        args.cross_lag_deltas,
    )
    if args.phi_top_k <= 0:
        F = fu.count_feature_columns(
            tx_n.shape[1],
            lags,
            args.orders_mem,
            args.orders_conj,
            args.use_conjugate,
            args.use_cross,
            cross_kinds=args.cross_kinds,
            cross_lag_deltas=args.cross_lag_deltas,
        )
        col_indices = np.arange(F, dtype=np.int64)
    else:
        from ridge_sic import _build_subset_matrix
        from task_and_baseline import MODEL_SUBSET

        X, F = _build_subset_matrix(
            tx_n,
            bp,
            lags,
            orders_mem=args.orders_mem,
            orders_conj=args.orders_conj,
            use_conjugate=args.use_conjugate,
            use_cross=args.use_cross,
            cross_kinds=args.cross_kinds,
            cross_lag_deltas=args.cross_lag_deltas,
            model_subset=MODEL_SUBSET,
        )
        sl = MODEL_SUBSET
        Y = np.column_stack(
            [
                bp(rx_res[:, ch]).astype(np.complex128, copy=False)[sl]
                for ch in range(4)
            ]
        )
        score = np.zeros(F, dtype=np.float64)
        for c in range(4):
            score += np.abs(X.conj().T @ Y[:, c])
        kk = min(int(args.phi_top_k), F)
        col_indices = np.argsort(-score)[:kk]
        del X
    return rx_res, I_t, col_indices


def _parse_int_tuple(s: str):
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def _parse_cross_kinds(s: str):
    if not s.strip():
        return fu.DEFAULT_CROSS_KINDS
    allowed = {
        fu.CROSS_KIND_IJ_MAG2,
        fu.CROSS_KIND_IJ_MAG4,
        fu.CROSS_KIND_CONJ_IJ_MAG2,
        fu.CROSS_KIND_CONJ_IJ_MAG4,
    }
    out = tuple(x.strip() for x in s.split(",") if x.strip())
    for x in out:
        if x not in allowed:
            raise ValueError(f"unknown cross kind {x!r}; allowed {allowed}")
    return out


def _parse_lag_deltas(s: str):
    if not s.strip():
        return fu.DEFAULT_CROSS_LAG_DELTAS
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mat", type=str, default="challenge.mat")
    p.add_argument("--checkpoint", type=str, default="feature_model.pt")
    p.add_argument("--device", type=str, default="")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--window_size", type=int, default=512)
    p.add_argument("--stride", type=int, default=256)
    p.add_argument("--k_left", type=int, default=-16)
    p.add_argument("--k_right", type=int, default=16)
    p.add_argument("--ridge_lambda", type=float, default=1e-6)
    p.add_argument("--orders_mem", type=str, default="1,3,5")
    p.add_argument("--orders_conj", type=str, default="1,3")
    p.add_argument("--use_conjugate", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--use_cross", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--cross_kinds", type=str, default="")
    p.add_argument("--cross_lag_deltas", type=str, default="")
    p.add_argument("--phi_top_k", type=int, default=384, help="<=0 means all columns")
    p.add_argument("--lambda_band", type=float, default=0.0)
    p.add_argument("--model", choices=["mlp", "cnn"], default="mlp")
    p.add_argument("--train_frac", type=float, default=0.85)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    args.orders_mem = _parse_int_tuple(args.orders_mem)
    args.orders_conj = _parse_int_tuple(args.orders_conj)
    args.cross_kinds = _parse_cross_kinds(args.cross_kinds)
    args.cross_lag_deltas = _parse_lag_deltas(args.cross_lag_deltas)

    device = pick_torch_device(args.device if args.device.strip() else None)
    configure_cuda_for_training(device)
    print(f"Device: {device}")

    tx_n, rx, Fs, N, helpers = load_data(args.mat)
    lags = list(range(args.k_left, args.k_right + 1))
    bp = make_bp_callable(helpers, Fs)

    rx_res, I_t, col_indices = _teacher_and_columns(tx_n, rx, helpers, Fs, lags, args, bp)
    in_dim = int(len(col_indices) * 2)
    print(f"Phi columns K={len(col_indices)}, in_dim={in_dim}")

    lag_abs = max(abs(min(lags)), abs(max(lags))) if lags else 0
    margin = lag_abs + max(abs(d) for d in args.cross_lag_deltas)
    w = args.window_size
    lo = margin
    hi = N - w - margin
    all_starts = list(range(lo, hi, args.stride))
    if not all_starts:
        raise RuntimeError(
            f"No training windows (N={N}, margin={margin}, window={w}). Reduce window or lags."
        )
    split = int(len(all_starts) * args.train_frac)
    train_starts = all_starts[:split]
    val_starts = all_starts[split:] or all_starts[-1:]

    train_ds = PhiWindowDataset(
        tx_n,
        rx_res,
        I_t,
        bp,
        lags,
        args.orders_mem,
        args.orders_conj,
        args.use_conjugate,
        args.use_cross,
        args.cross_kinds,
        args.cross_lag_deltas,
        col_indices,
        train_starts,
        w,
    )
    val_ds = PhiWindowDataset(
        tx_n,
        rx_res,
        I_t,
        bp,
        lags,
        args.orders_mem,
        args.orders_conj,
        args.use_conjugate,
        args.use_cross,
        args.cross_kinds,
        args.cross_lag_deltas,
        col_indices,
        val_starts,
        w,
    )

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
        model = FeatureMLPCanceller(in_dim=in_dim).to(device)
    else:
        model = FeatureCNNCanceller(in_dim=in_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best_val = float("inf")
    best_state = None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tr = 0.0
        ntr = 0
        for phi, tgt, rx_ri in train_loader:
            phi = phi.to(device, non_blocking=pin)
            tgt = tgt.to(device, non_blocking=pin)
            rx_ri = rx_ri.to(device, non_blocking=pin)
            pred = model(phi)
            mse = nn.functional.mse_loss(pred, tgt)
            loss = mse
            if args.lambda_band > 0:
                res = rx_ri - pred
                loss = loss + args.lambda_band * band_power_on_residual(res)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item() * phi.shape[0]
            ntr += phi.shape[0]
        sched.step()

        model.eval()
        va = 0.0
        nv = 0
        with torch.no_grad():
            for phi, tgt, _ in val_loader:
                phi = phi.to(device, non_blocking=pin)
                tgt = tgt.to(device, non_blocking=pin)
                pred = model(phi)
                va += nn.functional.mse_loss(pred, tgt).item() * phi.shape[0]
                nv += phi.shape[0]
        tr /= max(ntr, 1)
        va /= max(nv, 1)
        tag = ""
        if va < best_val:
            best_val = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = " *"
        print(f"epoch {epoch}/{args.epochs} train={tr:.6f} val={va:.6f}{tag}")

    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {
        "k_left": args.k_left,
        "k_right": args.k_right,
        "ridge_lambda": args.ridge_lambda,
        "orders_mem": list(args.orders_mem),
        "orders_conj": list(args.orders_conj),
        "use_conjugate": args.use_conjugate,
        "use_cross": args.use_cross,
        "cross_kinds": list(args.cross_kinds),
        "cross_lag_deltas": list(args.cross_lag_deltas),
        "phi_top_k": args.phi_top_k,
        "col_indices": col_indices.tolist(),
        "window_size": args.window_size,
        "model": args.model,
        "in_dim": in_dim,
        "Fs": Fs,
    }
    ckpt = {
        "state_dict": model.state_dict(),
        "meta": meta,
    }
    path = Path(args.checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {path} in {time.time()-t0:.1f}s, best_val_mse={best_val:.6f}")


if __name__ == "__main__":
    main()
