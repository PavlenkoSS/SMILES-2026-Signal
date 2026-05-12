"""Shared torch device selection and CUDA training tweaks."""
from __future__ import annotations

import torch


def pick_torch_device(device: str | None) -> torch.device:
    """Resolve torch device: explicit string, else cuda if available, else mps, else cpu."""
    if device is not None and str(device).strip():
        d = torch.device(str(device).strip())
        if d.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        if d.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        return d
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_cuda_for_training(device: torch.device) -> None:
    """No-op on CPU/MPS. On CUDA: cuDNN autotune + TF32 matmul for speed on Ampere+."""
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Safe default for most training; faster on Ampere+
