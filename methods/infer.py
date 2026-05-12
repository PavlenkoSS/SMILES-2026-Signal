"""Chunked overlap-add inference for full-signal prediction."""
import argparse

import numpy as np
import torch

from .data_utils import load_data, complex_to_real, real_to_complex
from .device_utils import pick_torch_device
from .models import MLPInterferenceCanceller, CNNGRUCanceller, TransformerInterferenceCanceller

pick_inference_device = pick_torch_device  # backward-compatible name


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

            chunk = torch.from_numpy(chunk_data).unsqueeze(0).to(
                device, non_blocking=(device.type == "cuda")
            )
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


def chunked_feature_inference(
    model,
    tx_n,
    bp,
    lags,
    col_indices,
    orders_mem,
    orders_conj,
    use_conjugate,
    use_cross,
    cross_kinds,
    cross_lag_deltas,
    window_size,
    overlap,
    device,
):
    """Overlap-add neural interference (real 8-ch) from Phi windows."""
    from phi_features import phi_window_real

    N = tx_n.shape[0]
    pred_sum = np.zeros((N, 8), dtype=np.float32)
    weight_sum = np.zeros((N, 1), dtype=np.float32)
    step = max(1, window_size - overlap)

    model.eval()
    with torch.no_grad():
        for start in range(0, N, step):
            end = min(start + window_size, N)
            actual = end - start
            if actual < 32:
                break
            phi = phi_window_real(
                tx_n,
                bp,
                lags,
                orders_mem,
                orders_conj,
                use_conjugate,
                use_cross,
                cross_kinds,
                cross_lag_deltas,
                col_indices,
                start,
                actual,
            )
            if actual < window_size:
                pad = window_size - actual
                phi = np.pad(phi, ((0, pad), (0, 0)), mode="constant")

            chunk = torch.from_numpy(phi).unsqueeze(0).to(
                device, non_blocking=(device.type == "cuda")
            )
            pred = model(chunk).squeeze(0).cpu().numpy()

            w = np.hanning(actual).reshape(-1, 1).astype(np.float32)
            pred_sum[start:end] += pred[:actual] * w
            weight_sum[start:end] += w

    pred_out = pred_sum / (weight_sum + 1e-8)
    return real_to_complex(pred_out)


def load_and_infer_feature(
    tx_n,
    rx,
    checkpoint,
    helpers,
    fs_hz,
    device=None,
    overlap=128,
    ridge_blend=0.0,
):
    """Phi-only neural canceller + optional blend with ridge teacher (0=pure neural I)."""
    from ridge_sic import fit_ridge_weights, predict_poly_from_weights
    from sic_filters import make_bp_callable

    if device is None or (isinstance(device, str) and not device.strip()):
        device = pick_inference_device(None)
    elif isinstance(device, str):
        device = pick_inference_device(device)
    else:
        device = torch.device(device)

    try:
        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint, map_location=device)
    meta = ckpt["meta"]
    lags = list(range(int(meta["k_left"]), int(meta["k_right"]) + 1))
    col_indices = np.asarray(meta["col_indices"], dtype=np.int64)
    orders_mem = tuple(meta["orders_mem"])
    orders_conj = tuple(meta["orders_conj"])
    use_conjugate = bool(meta["use_conjugate"])
    use_cross = bool(meta["use_cross"])
    cross_kinds = tuple(meta["cross_kinds"])
    cross_lag_deltas = tuple(int(x) for x in meta["cross_lag_deltas"])
    window_size = int(meta["window_size"])
    model_type = meta.get("model", "mlp")
    in_dim = int(meta["in_dim"])

    from models_feature import FeatureCNNCanceller, FeatureMLPCanceller

    if model_type == "cnn":
        model = FeatureCNNCanceller(in_dim=in_dim)
    else:
        model = FeatureMLPCanceller(in_dim=in_dim)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    bp = make_bp_callable(helpers, fs_hz)
    fit_tx = helpers["fit_tx_prediction"]
    tx_pred = fit_tx(rx)
    rx_res = rx - tx_pred

    I_nn = chunked_feature_inference(
        model,
        tx_n,
        bp,
        lags,
        col_indices,
        orders_mem,
        orders_conj,
        use_conjugate,
        use_cross,
        cross_kinds,
        cross_lag_deltas,
        window_size,
        overlap,
        device,
    )

    if ridge_blend > 0.0:
        W, _, _ = fit_ridge_weights(
            tx_n,
            rx_res,
            helpers,
            lags,
            fs_hz,
            ridge_lambda=float(meta["ridge_lambda"]),
            orders_mem=orders_mem,
            orders_conj=orders_conj,
            use_conjugate=use_conjugate,
            use_cross=use_cross,
            cross_kinds=cross_kinds,
            cross_lag_deltas=cross_lag_deltas,
        )
        I_ridge = predict_poly_from_weights(
            tx_n,
            bp,
            lags,
            W,
            orders_mem,
            orders_conj,
            use_conjugate,
            use_cross,
            cross_kinds,
            cross_lag_deltas,
        )
        I_use = (1.0 - ridge_blend) * I_nn + ridge_blend * I_ridge
    else:
        I_use = I_nn

    return rx - tx_pred - I_use


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
