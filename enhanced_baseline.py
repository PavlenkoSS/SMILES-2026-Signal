"""Enhanced baseline: official baseline + rank-1 external interference removal.

The official baseline only removes TX-dependent interference but NOT the
rank-1 spatially coherent external component. Adding rank-1 removal should
improve the score while keeping validity (the scorer already accounts for it).
"""
import numpy as np
from scipy.signal import convolve, firwin

from data_utils import load_data
from task_and_baseline import baseline, build_task_helpers


CENTER = 1.9e6
BW = 0.6e6


def make_bandpass(center_hz, bw_hz, fs_hz, n_taps=2047):
    lp = firwin(n_taps, bw_hz / 2, window="blackman", fs=fs_hz)
    return lp * np.exp(2j * np.pi * center_hz / fs_hz * np.arange(n_taps))


def rank1_from_band_matrix(band_matrix):
    """Extract rank-1 spatially coherent component via eigendecomposition."""
    cov = band_matrix.conj().T @ band_matrix / band_matrix.shape[0]
    _, vecs = np.linalg.eigh(cov)
    shared = band_matrix @ vecs[:, -1]  # project onto dominant eigenvector
    denom = np.vdot(shared, shared) + 1e-30
    return np.column_stack(
        [(np.vdot(shared, band_matrix[:, ch]) / denom) * shared
         for ch in range(band_matrix.shape[1])]
    )


def enhanced_canceller(tx_n, rx, helpers, lambda_rank1=1.0):
    """Baseline + conservative rank-1 removal.
    
    lambda_rank1: scaling for rank-1 subtraction [0, 1].
    """
    fit_tx = helpers["fit_tx_prediction"]
    score_filter = helpers["score_filter"]

    # Standard baseline: subtract TX-dependent interference
    tx_pred = fit_tx(rx)
    rx_after_bl = rx - tx_pred

    if lambda_rank1 <= 0.0:
        return rx_after_bl

    # Compute bandpass-filtered residual
    residual_band = np.column_stack(
        [score_filter(rx_after_bl[:, ch]) for ch in range(rx_after_bl.shape[1])]
    )

    # Extract rank-1 component
    rank1 = rank1_from_band_matrix(residual_band)

    # Subtract scaled rank-1 component
    rx_hat = rx_after_bl - lambda_rank1 * rank1

    return rx_hat


def search_lambda(tx_n, rx, helpers, lambdas=None):
    """Search for the best lambda_rank1 that maximizes valid score."""
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    best_avg = -999
    best_lam = 0.0

    for lam in lambdas:
        rx_hat = enhanced_canceller(tx_n, rx, helpers, lambda_rank1=lam)
        print(f"\n--- lambda={lam:.2f} ---")
        reds, avg = helpers["score"](rx, rx_hat, label=f"lam={lam:.2f}")
        if avg > best_avg:
            best_avg = avg
            best_lam = lam

    print(f"\n=== Best: lambda={best_lam:.2f}, avg={best_avg:.4f} dB ===")
    return best_lam, best_avg


if __name__ == "__main__":
    print("Loading data...")
    tx_n, rx, Fs, N, helpers = load_data()

    print("\n=== Standard Baseline ===")
    rx_bl = baseline(tx_n, rx, helpers["fit_tx_prediction"])
    helpers["score"](rx, rx_bl, label="baseline")

    print("\n=== Searching lambda for rank-1 enhancement ===")
    best_lam, best_avg = search_lambda(tx_n, rx, helpers,
                                        lambdas=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
