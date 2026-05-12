"""Overlap-add ridge: per-block fit on bandpassed targets, Hann blend of I_tx."""
import numpy as np

from . import feature_utils as fu
from .ridge_sic import fit_ridge_weights_region, predict_poly_from_weights
from .sic_filters import make_bp_callable


def blockwise_canceller(
    tx_n,
    rx,
    helpers,
    fs_hz,
    lags,
    ridge_lambda=1e-5,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
    cross_kinds=fu.DEFAULT_CROSS_KINDS,
    cross_lag_deltas=fu.DEFAULT_CROSS_LAG_DELTAS,
    block_size=131072,
    overlap_frac=0.5,
):
    fit_tx = helpers["fit_tx_prediction"]
    bp = make_bp_callable(helpers, fs_hz)
    tx_pred = fit_tx(rx)
    rx_res = rx - tx_pred

    n_samples = tx_n.shape[0]
    hop = max(1, int(block_size * (1.0 - float(overlap_frac))))

    I_sum = np.zeros((n_samples, 4), dtype=np.complex128)
    w_sum = np.zeros((n_samples, 1), dtype=np.float64)

    b0 = 0
    while b0 < n_samples:
        b1 = min(b0 + block_size, n_samples)
        train_slice = slice(b0, b1)
        W, _, _ = fit_ridge_weights_region(
            tx_n,
            rx_res,
            helpers,
            lags,
            fs_hz,
            train_slice,
            ridge_lambda=ridge_lambda,
            orders_mem=orders_mem,
            orders_conj=orders_conj,
            use_conjugate=use_conjugate,
            use_cross=use_cross,
            cross_kinds=cross_kinds,
            cross_lag_deltas=cross_lag_deltas,
        )
        poly_full = predict_poly_from_weights(
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
        seg_len = b1 - b0
        w = np.hanning(seg_len).reshape(-1, 1).astype(np.float64)
        I_sum[b0:b1] += poly_full[b0:b1] * w
        w_sum[b0:b1] += w
        if b1 >= n_samples:
            break
        b0 += hop

    I = I_sum / (w_sum + 1e-30)
    return rx_res - I
