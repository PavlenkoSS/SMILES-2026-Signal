"""Rank-1 SVD subtraction on bandpass residual (matches scorer geometry)."""
import contextlib
import io

import numpy as np

from .sic_filters import scoring_bandpass_kernel


def rank1_from_band_matrix(band_matrix):
    cov = band_matrix.conj().T @ band_matrix / band_matrix.shape[0]
    _, vecs = np.linalg.eigh(cov)
    shared = band_matrix @ vecs[:, -1]
    denom = np.vdot(shared, shared) + 1e-30
    return np.column_stack(
        [
            (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
            for ch in range(band_matrix.shape[1])
        ]
    )


def bandpass_residual_matrix(rx_partial, score_filter, n_ch=4):
    return np.column_stack(
        [score_filter(rx_partial[:, ch]) for ch in range(n_ch)]
    )


def subtract_rank1(rx_partial, helpers, alpha=1.0):
    score_filter = helpers["score_filter"]
    rb = bandpass_residual_matrix(rx_partial, score_filter)
    r1 = rank1_from_band_matrix(rb)
    return rx_partial - alpha * r1


def score_silent(helpers, rx_before, rx_after, label=""):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return helpers["score"](rx_before, rx_after, label=label)


def grid_search_rank1_alpha(rx_before, rx_partial, helpers, alphas=None):
    if alphas is None:
        alphas = [round(a, 2) for a in np.arange(0.0, 1.01, 0.05)]
    score_filter = helpers["score_filter"]
    rb = bandpass_residual_matrix(rx_partial, score_filter)
    r1 = rank1_from_band_matrix(rb)
    best = None
    best_avg = -1e9
    best_alpha = 0.0
    for a in alphas:
        rx_hat = rx_partial - float(a) * r1
        _, avg = score_silent(helpers, rx_before, rx_hat, label=f"rank1 a={a:.2f}")
        if avg > best_avg:
            best_avg = avg
            best = rx_hat.copy()
            best_alpha = float(a)
    return best, best_alpha, best_avg


# ---------------------------------------------------------------------------
# Band preimage: find raw waveform c such that score_filter(c) = D
# ---------------------------------------------------------------------------

def _bp_kernel_fft(fs_hz, N):
    """FFT of scorer bandpass kernel, zero-padded to length N."""
    kernel = scoring_bandpass_kernel(fs_hz)
    return np.fft.fft(kernel, n=N)


def band_preimage(D, fs_hz, reg=1e-8):
    """Per-channel Wiener deconvolution: find C such that bandpass(C) ≈ D.

    D: (N,) or (N, n_ch) complex — desired band-domain signal.
    Returns C with same shape. Only inverts inside the passband; zero outside.
    """
    if D.ndim == 1:
        return _band_preimage_1d(D, fs_hz, reg)
    out = np.zeros_like(D)
    for ch in range(D.shape[1]):
        out[:, ch] = _band_preimage_1d(D[:, ch], fs_hz, reg)
    return out


def _band_preimage_1d(d, fs_hz, reg=1e-8):
    N = len(d)
    H = _bp_kernel_fft(fs_hz, N)
    D_f = np.fft.fft(d)
    H_mag2 = np.abs(H) ** 2
    thr = reg * np.max(H_mag2)
    mask = H_mag2 > thr
    C = np.zeros(N, dtype=np.complex128)
    C[mask] = D_f[mask] * H[mask].conj() / (H_mag2[mask] + thr)
    return np.fft.ifft(C)


def grid_search_band_inverse(rx, helpers, fs_hz, alphas=None):
    """Scorer-aligned rank-1 via band_preimage.

    1. Subtract TX interference via the official baseline (fit_tx_prediction).
    2. Compute rank-1 on the bandpass residual.
    3. Subtract band_preimage(alpha * rank1) from the post-baseline signal.

    The scorer will see removed = tx_pred + alpha*rank1_preimage, which
    decomposes cleanly into TX + rank-1.
    """
    if alphas is None:
        alphas = np.concatenate([
            np.linspace(0.0, 1.1, 56),
            np.linspace(1.1, 2.0, 19),
        ])

    score_filter = helpers["score_filter"]
    fit_tx = helpers["fit_tx_prediction"]

    tx_pred = fit_tx(rx)
    rx_bl = rx - tx_pred

    R_bl = bandpass_residual_matrix(rx_bl, score_filter)
    D_r1 = rank1_from_band_matrix(R_bl)

    r1_preimage = band_preimage(D_r1, fs_hz)

    best = None
    best_avg = -1e9
    best_alpha = 0.0
    for alpha in alphas:
        rx_hat = rx_bl - float(alpha) * r1_preimage
        _, avg = score_silent(helpers, rx, rx_hat, label=f"band-inv a={alpha:.3f}")
        if avg > best_avg:
            best_avg = avg
            best = rx_hat.copy()
            best_alpha = float(alpha)
    return best, best_alpha, best_avg


# ---------------------------------------------------------------------------
# Legacy wideband rank-1 (kept for backwards compat)
# ---------------------------------------------------------------------------

def grid_search_wideband_rank1(rx_before, rx_partial, helpers, fs_hz, alphas=None):
    """Grid-search alpha for wideband rank-1 subtraction."""
    if alphas is None:
        alphas = [round(a, 2) for a in np.arange(0.0, 1.01, 0.05)]

    score_filter = helpers["score_filter"]
    rb = bandpass_residual_matrix(rx_partial, score_filter)
    r1 = rank1_from_band_matrix(rb)

    wb_corr = band_preimage(r1, fs_hz)

    best = None
    best_avg = -1e9
    best_alpha = 0.0
    for a in alphas:
        rx_hat = rx_partial - float(a) * wb_corr
        _, avg = score_silent(helpers, rx_before, rx_hat, label=f"wb_rank1 a={a:.2f}")
        if avg > best_avg:
            best_avg = avg
            best = rx_hat.copy()
            best_alpha = float(a)
    return best, best_alpha, best_avg
