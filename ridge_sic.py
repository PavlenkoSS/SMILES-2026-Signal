"""Chunked complex ridge regression for TX-based SIC on the scorer subset."""
import contextlib
import io

import numpy as np

from task_and_baseline import MODEL_SUBSET
import feature_utils as fu
from sic_filters import make_bp_callable


def _stack_targets(rx_res, bp, model_subset=MODEL_SUBSET):
    sl = model_subset
    return np.column_stack(
        [
            bp(rx_res[:, ch]).astype(np.complex128, copy=False)[sl]
            for ch in range(4)
        ]
    )


def _build_subset_matrix(
    tx_n,
    bp,
    lags,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
    model_subset=MODEL_SUBSET,
):
    raw_lists = []
    raw_lists.append(fu.build_memory_polynomial_features(tx_n, lags, orders=orders_mem))
    if use_conjugate:
        raw_lists.append(fu.build_conjugate_features(tx_n, lags, orders=orders_conj))
    if use_cross:
        raw_lists.append(fu.build_cross_channel_features(tx_n, lags))
    raw_feats = [f for group in raw_lists for f in group]

    sl = model_subset
    n_sub = sl.stop - sl.start
    n_feat = len(raw_feats)
    X = np.zeros((n_sub, n_feat), dtype=np.complex128)

    for j, raw in enumerate(raw_feats):
        filt = bp(raw.astype(np.complex128, copy=False))
        X[:, j] = filt[sl]
        del raw
        del filt

    return X, n_feat


def _raw_feature_groups(
    tx_n,
    lags,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
):
    raw_lists = []
    raw_lists.append(fu.build_memory_polynomial_features(tx_n, lags, orders=orders_mem))
    if use_conjugate:
        raw_lists.append(fu.build_conjugate_features(tx_n, lags, orders=orders_conj))
    if use_cross:
        raw_lists.append(fu.build_cross_channel_features(tx_n, lags))
    return raw_lists


def _solve_ridge(G, B, lam):
    F = G.shape[0]
    reg = lam * np.eye(F, dtype=np.float64)
    G_reg = G + reg
    return np.linalg.solve(G_reg, B)


def predict_poly_from_weights(
    tx_n,
    bp,
    lags,
    W,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
):
    preds, = predict_poly_from_weights_batch(
        tx_n,
        bp,
        lags,
        [W],
        orders_mem=orders_mem,
        orders_conj=orders_conj,
        use_conjugate=use_conjugate,
        use_cross=use_cross,
    )
    return preds


def predict_poly_from_weights_batch(
    tx_n,
    bp,
    lags,
    W_list,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
):
    raw_feats = [
        f
        for group in _raw_feature_groups(
            tx_n, lags, orders_mem, orders_conj, use_conjugate, use_cross
        )
        for f in group
    ]
    n_samples = tx_n.shape[0]
    preds = [
        np.zeros((n_samples, 4), dtype=np.complex128) for _ in range(len(W_list))
    ]
    for j, raw in enumerate(raw_feats):
        col = bp(raw.astype(np.complex128, copy=False))
        for wi, W in enumerate(W_list):
            preds[wi] += np.outer(col, W[j, :])
        del raw
        del col
    return preds


def fit_ridge_weights(
    tx_n,
    rx_residual,
    helpers,
    lags,
    fs_hz,
    ridge_lambda=1e-6,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
):
    """Fit W on MODEL_SUBSET with target = bandpass(rx_residual)."""
    bp = make_bp_callable(helpers, fs_hz)
    X, _ = _build_subset_matrix(
        tx_n,
        bp,
        lags,
        orders_mem=orders_mem,
        orders_conj=orders_conj,
        use_conjugate=use_conjugate,
        use_cross=use_cross,
    )
    Y = _stack_targets(rx_residual, bp)
    G = X.conj().T @ X
    B = X.conj().T @ Y
    del X
    W = _solve_ridge(G, B, float(ridge_lambda))
    return W, G, B


def ridge_canceller(
    tx_n,
    rx,
    helpers,
    lags,
    fs_hz,
    ridge_lambda=1e-6,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
):
    fit_tx = helpers["fit_tx_prediction"]
    bp = make_bp_callable(helpers, fs_hz)
    tx_pred = fit_tx(rx)
    rx_res = rx - tx_pred
    W, _, _ = fit_ridge_weights(
        tx_n,
        rx_res,
        helpers,
        lags,
        fs_hz,
        ridge_lambda=ridge_lambda,
        orders_mem=orders_mem,
        orders_conj=orders_conj,
        use_conjugate=use_conjugate,
        use_cross=use_cross,
    )
    poly_pred = predict_poly_from_weights(
        tx_n,
        bp,
        lags,
        W,
        orders_mem,
        orders_conj,
        use_conjugate,
        use_cross,
    )
    return rx_res - poly_pred


def score_silent(helpers, rx_before, rx_after, label=""):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return helpers["score"](rx_before, rx_after, label=label)


def grid_search_ridge_lambda(
    tx_n,
    rx,
    helpers,
    lags,
    fs_hz,
    lambdas=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3),
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
):
    fit_tx = helpers["fit_tx_prediction"]
    bp = make_bp_callable(helpers, fs_hz)
    tx_pred = fit_tx(rx)
    rx_res = rx - tx_pred

    X, _ = _build_subset_matrix(
        tx_n,
        bp,
        lags,
        orders_mem=orders_mem,
        orders_conj=orders_conj,
        use_conjugate=use_conjugate,
        use_cross=use_cross,
    )
    Y = _stack_targets(rx_res, bp)
    G = X.conj().T @ X
    B = X.conj().T @ Y
    del X

    W_list = [_solve_ridge(G, B, float(lam)) for lam in lambdas]
    preds = predict_poly_from_weights_batch(
        tx_n,
        bp,
        lags,
        W_list,
        orders_mem=orders_mem,
        orders_conj=orders_conj,
        use_conjugate=use_conjugate,
        use_cross=use_cross,
    )

    best_rx = None
    best_avg = -1e9
    best_lam = float(lambdas[0])

    for lam, poly_pred in zip(lambdas, preds):
        rx_hat = rx_res - poly_pred
        _, avg = score_silent(helpers, rx, rx_hat, label=f"ridge lam={lam:.1e}")
        if avg > best_avg:
            best_avg = avg
            best_rx = rx_hat.copy()
            best_lam = float(lam)

    return best_rx, best_lam, best_avg
