"""Rank-1 SVD subtraction on bandpass residual (matches scorer geometry)."""
import contextlib
import io

import numpy as np


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
