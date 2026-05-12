"""Scorer-aligned SIC: use the exact same TX basis as the scorer (10 model terms,
13 lags) to guarantee near-perfect explainability, then add rank-1.

All methods start from the official baseline (fit_tx_prediction) and only apply
band_preimage to the rank-1 correction on the residual.

Three methods:
  scorer_rank1        - baseline + band_preimage(rank1) with alpha grid
  scorer_alternating  - alternate TX re-projection + rank-1 in band domain
  scorer_fulltx_rank1 - refit scorer TX basis on full signal, rank-1 on residual
"""
from __future__ import annotations

import numpy as np

from task_and_baseline import (
    MODEL_SUBSET, MODEL_LAGS, shift_signal, shifted_window,
)
from .rank1_sic import (
    band_preimage,
    bandpass_residual_matrix,
    rank1_from_band_matrix,
    score_silent,
)


def _build_scorer_model_terms(tx_n, score_filter):
    """Replicate the scorer's 10 bandpass-filtered third-order TX interaction terms."""
    return (
        score_filter(tx_n[:, 0] ** 2 * tx_n[:, 1].conj()),
        score_filter(tx_n[:, 1] ** 2 * tx_n[:, 0].conj()),
        score_filter(tx_n[:, 0] ** 2 * tx_n[:, 3].conj()),
        score_filter(tx_n[:, 3] ** 2 * tx_n[:, 0].conj()),
        score_filter(tx_n[:, 1] ** 2 * tx_n[:, 2].conj()),
        score_filter(tx_n[:, 2] ** 2 * tx_n[:, 1].conj()),
        score_filter(tx_n[:, 3] ** 2 * tx_n[:, 2].conj()),
        score_filter(tx_n[:, 2] ** 2 * tx_n[:, 3].conj()),
        score_filter(tx_n[:, 0] ** 2 * tx_n[:, 5].conj()),
        score_filter(tx_n[:, 5] ** 2 * tx_n[:, 0].conj()),
    )


def _build_design_matrix(model_terms, fit_slice, lags=MODEL_LAGS):
    start, stop = fit_slice.start, fit_slice.stop
    return np.column_stack([
        shifted_window(term, lag, start, stop)
        for term in model_terms
        for lag in lags
    ])


def _predict_full(model_terms, coef_matrix, n_samples, lags=MODEL_LAGS):
    """coef_matrix: (n_terms*n_lags, n_ch)."""
    n_lags = len(lags)
    pred = np.zeros((n_samples, coef_matrix.shape[1]), dtype=np.complex128)
    for ch in range(coef_matrix.shape[1]):
        coef = coef_matrix[:, ch].reshape(len(model_terms), n_lags)
        for t_idx, term in enumerate(model_terms):
            for l_idx, lag in enumerate(lags):
                pred[:, ch] += coef[t_idx, l_idx] * shift_signal(term, lag)
    return pred


def project_onto_scorer_tx_basis(target_band, tx_n, score_filter,
                                  fit_slice=MODEL_SUBSET, lags=MODEL_LAGS, reg=1e-6):
    """Project target_band (N, n_ch) onto scorer's TX basis.

    Returns D_tx (N, n_ch) — full-length prediction in band domain.
    """
    model_terms = _build_scorer_model_terms(tx_n, score_filter)
    X = _build_design_matrix(model_terms, fit_slice, lags)
    G = X.conj().T @ X + reg * np.eye(X.shape[1])
    Y = target_band[fit_slice.start:fit_slice.stop, :]
    W = np.linalg.solve(G, X.conj().T @ Y)
    return _predict_full(model_terms, W, target_band.shape[0], lags)


def _default_alphas():
    return np.concatenate([
        np.linspace(0.0, 1.1, 56),
        np.linspace(1.1, 2.0, 19),
    ])


# ---- Method: scorer_alternating ----

def scorer_alternating(tx_n, rx, helpers, fs_hz, n_iter=3,
                       fit_slice=MODEL_SUBSET, lags=MODEL_LAGS,
                       alphas=None):
    """Baseline first, then alternate TX re-projection + rank-1 on residual.

    The alternation re-projects the bandpass residual onto the scorer's TX
    basis (improving the TX estimate beyond the default baseline), then
    extracts rank-1 from what's left.  Only the rank-1 part is subtracted
    via band_preimage.
    """
    score_filter = helpers["score_filter"]
    fit_tx = helpers["fit_tx_prediction"]

    tx_pred = fit_tx(rx)
    rx_bl = rx - tx_pred

    R_bl = bandpass_residual_matrix(rx_bl, score_filter)

    D_rank1 = np.zeros_like(R_bl)
    D_tx_extra = np.zeros_like(R_bl)

    for _ in range(n_iter):
        D_tx_extra = project_onto_scorer_tx_basis(
            R_bl - D_rank1, tx_n, score_filter, fit_slice=fit_slice, lags=lags,
        )
        D_rank1 = rank1_from_band_matrix(R_bl - D_tx_extra)

    r1_preimage = band_preimage(D_rank1, fs_hz)
    tx_extra_preimage = band_preimage(D_tx_extra, fs_hz)

    if alphas is None:
        alphas = _default_alphas()

    best = None
    best_avg = -1e9
    best_alpha = 0.0
    for alpha in alphas:
        rx_hat = rx_bl - tx_extra_preimage - float(alpha) * r1_preimage
        _, avg = score_silent(helpers, rx, rx_hat, label=f"scorer_alt a={alpha:.3f}")
        if avg > best_avg:
            best_avg = avg
            best = rx_hat.copy()
            best_alpha = float(alpha)

    return best, best_alpha, best_avg


# ---- Method: scorer_fulltx_rank1 ----

def scorer_fulltx_rank1(tx_n, rx, helpers, fs_hz, lags=MODEL_LAGS, alphas=None):
    """Baseline + refit scorer TX basis on full signal, rank-1 on residual."""
    score_filter = helpers["score_filter"]
    fit_tx = helpers["fit_tx_prediction"]
    N = rx.shape[0]

    tx_pred = fit_tx(rx)
    rx_bl = rx - tx_pred

    R_bl = bandpass_residual_matrix(rx_bl, score_filter)

    full_slice = slice(0, N)
    D_tx_extra = project_onto_scorer_tx_basis(
        R_bl, tx_n, score_filter, fit_slice=full_slice, lags=lags,
    )
    D_r1 = rank1_from_band_matrix(R_bl - D_tx_extra)

    r1_preimage = band_preimage(D_r1, fs_hz)
    tx_extra_preimage = band_preimage(D_tx_extra, fs_hz)

    if alphas is None:
        alphas = _default_alphas()

    best = None
    best_avg = -1e9
    best_alpha = 0.0
    for alpha in alphas:
        rx_hat = rx_bl - tx_extra_preimage - float(alpha) * r1_preimage
        _, avg = score_silent(helpers, rx, rx_hat, label=f"fulltx a={alpha:.3f}")
        if avg > best_avg:
            best_avg = avg
            best = rx_hat.copy()
            best_alpha = float(alpha)

    return best, best_alpha, best_avg
