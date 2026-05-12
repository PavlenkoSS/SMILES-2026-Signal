"""Chunked overlap-add inference for full-signal prediction."""
import argparse

import numpy as np
import torch

from data_utils import load_data, complex_to_real, real_to_complex
from models import MLPInterferenceCanceller, CNNGRUCanceller, TransformerInterferenceCanceller


def pick_inference_device(device: str | None) -> torch.device:
    """Resolve torch device for inference: CUDA preferred when available."""
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


def chunked_inference(model, tx_real, chunk_size=8192, overlap=256, device=None, pad_to_chunk=False):
    """Run model over full signal with Hann-windowed overlap-add."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    N = len(tx_real)
    out_ch = 8
    pred_sum = np.zeros((N, out_ch), dtype=np.float32)
    weight_sum = np.zeros((N, 1), dtype=np.float32)
    step = chunk_size - overlap

    with torch.no_grad():
        for start in range(0, N, step):
            end = min(start + chunk_size, N)
            actual_len = end - start
            if actual_len < 32:
                break

            chunk_data = tx_real[start:end].copy()
            if pad_to_chunk and actual_len < chunk_size:
                pad_len = chunk_size - actual_len
                chunk_data = np.pad(chunk_data, ((0, pad_len), (0, 0)), mode='constant')

            chunk = torch.from_numpy(chunk_data).unsqueeze(0).to(device)
            pred = model(chunk).squeeze(0).cpu().numpy()

            w = np.hanning(actual_len).reshape(-1, 1).astype(np.float32)
            pred_sum[start:end] += pred[:actual_len] * w
            weight_sum[start:end] += w

    return pred_sum / (weight_sum + 1e-8)


def load_and_infer(tx_n, rx, checkpoint="best_model.pt", model_type="cnngru",
                   chunk_size=8192, overlap=256, device=None, blend_baseline=0.0,
                   helpers=None):
    """Full pipeline: load checkpoint, run chunked inference, return rx_hat.
    
    blend_baseline: float in [0, 1]. If > 0, blend neural prediction with baseline:
        final_interference = (1-blend)*neural + blend*baseline_interference

    device: torch.device or str. If None, uses CUDA if available, else MPS, else CPU.
    """
    if device is None or (isinstance(device, str) and not device.strip()):
        device = pick_inference_device(None)
    elif isinstance(device, str):
        device = pick_inference_device(device)
    else:
        device = torch.device(device)

    if model_type == "mlp":
        model = MLPInterferenceCanceller(window_size=chunk_size)
        pad_to_chunk = True
    elif model_type == "transformer":
        model = TransformerInterferenceCanceller()
        pad_to_chunk = False
    else:
        model = CNNGRUCanceller()
        pad_to_chunk = False

    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    tx_real = complex_to_real(tx_n)  # (N, 12)
    pred_real = chunked_inference(model, tx_real, chunk_size=chunk_size, overlap=overlap,
                                  device=device, pad_to_chunk=pad_to_chunk)
    pred_complex = real_to_complex(pred_real)  # (N, 4)

    if blend_baseline > 0.0 and helpers is not None:
        from task_and_baseline import baseline
        baseline_interference = helpers["fit_tx_prediction"](rx)
        pred_complex = (1.0 - blend_baseline) * pred_complex + blend_baseline * baseline_interference

    rx_hat = rx - pred_complex
    return rx_hat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--model", choices=["mlp", "cnngru", "transformer"], default="cnngru")
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--blend", type=float, default=0.0,
                        help="Blend factor with baseline (0=pure neural, 1=pure baseline)")
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Torch device: cuda, cuda:0, cpu, mps, or empty for auto (cuda>mps>cpu)",
    )
    args = parser.parse_args()

    print("Loading data...")
    tx_n, rx, Fs, N, helpers = load_data()

    dev = pick_inference_device(args.device if args.device.strip() else None)
    print(f"Using device: {dev}")

    print(f"Running inference (blend={args.blend})...")
    rx_hat = load_and_infer(tx_n, rx, checkpoint=args.checkpoint, model_type=args.model,
                            chunk_size=args.chunk_size, overlap=args.overlap,
                            blend_baseline=args.blend, helpers=helpers, device=dev)

    print("\n=== Neural Model Score ===")
    reds, avg = helpers["score"](rx, rx_hat, label="neural")
    print(f"Average: {avg:.4f} dB")

    print("\n=== Baseline Score ===")
    from task_and_baseline import baseline
    rx_bl = baseline(tx_n, rx, helpers["fit_tx_prediction"])
    bl_reds, bl_avg = helpers["score"](rx, rx_bl, label="baseline")
    print(f"Baseline average: {bl_avg:.4f} dB")


if __name__ == "__main__":
    main()
