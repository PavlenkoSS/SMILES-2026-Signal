"""Chunked complex ridge regression for TX-based SIC on the scorer subset."""
import contextlib
import io

import numpy as np

from task_and_baseline import MODEL_SUBSET

from . import feature_utils as fu
from .sic_filters import make_bp_callable

_DEFAULT_CROSS = (fu.DEFAULT_CROSS_KINDS, fu.DEFAULT_CROSS_LAG_DELTAS)


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
    cross_kinds=fu.DEFAULT_CROSS_KINDS,
    cross_lag_deltas=fu.DEFAULT_CROSS_LAG_DELTAS,
    model_subset=MODEL_SUBSET,
):
    n_tx = tx_n.shape[1]
    n_feat = fu.count_feature_columns(
        n_tx,
        lags,
        orders_mem,
        orders_conj,
        use_conjugate,
        use_cross,
        cross_kinds=cross_kinds,
        cross_lag_deltas=cross_lag_deltas,
    )
    sl = model_subset
    n_sub = sl.stop - sl.start
    X = np.zeros((n_sub, n_feat), dtype=np.complex128)

    for j, raw in enumerate(
        fu.iter_all_raw_features(
            tx_n,
            lags,
            orders_mem,
            orders_conj,
            use_conjugate,
            use_cross,
            cross_kinds=cross_kinds,
            cross_lag_deltas=cross_lag_deltas,
        )
    ):
        filt = bp(raw.astype(np.complex128, copy=False))
        X[:, j] = filt[sl]
        del raw
        del filt

    return X, n_feat


def predict_poly_from_weights_batch(
    tx_n,
    bp,
    lags,
    W_list,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
    cross_kinds=fu.DEFAULT_CROSS_KINDS,
    cross_lag_deltas=fu.DEFAULT_CROSS_LAG_DELTAS,
):
    n_samples = tx_n.shape[0]
    preds = [
        np.zeros((n_samples, 4), dtype=np.complex128) for _ in range(len(W_list))
    ]
    for j, raw in enumerate(
        fu.iter_all_raw_features(
            tx_n,
            lags,
            orders_mem,
            orders_conj,
            use_conjugate,
            use_cross,
            cross_kinds=cross_kinds,
            cross_lag_deltas=cross_lag_deltas,
        )
    ):
        col = bp(raw.astype(np.complex128, copy=False))
        for wi, W in enumerate(W_list):
            preds[wi] += np.outer(col, W[j, :])
        del raw
        del col
    return preds


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
    cross_kinds=fu.DEFAULT_CROSS_KINDS,
    cross_lag_deltas=fu.DEFAULT_CROSS_LAG_DELTAS,
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
        cross_kinds=cross_kinds,
        cross_lag_deltas=cross_lag_deltas,
    )
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
    cross_kinds=fu.DEFAULT_CROSS_KINDS,
    cross_lag_deltas=fu.DEFAULT_CROSS_LAG_DELTAS,
    model_subset=MODEL_SUBSET,
):
    """Fit W on model_subset with target = bandpass(rx_residual)."""
    bp = make_bp_callable(helpers, fs_hz)
    X, _ = _build_subset_matrix(
        tx_n,
        bp,
        lags,
        orders_mem=orders_mem,
        orders_conj=orders_conj,
        use_conjugate=use_conjugate,
        use_cross=use_cross,
        cross_kinds=cross_kinds,
        cross_lag_deltas=cross_lag_deltas,
        model_subset=model_subset,
    )
    Y = _stack_targets(rx_residual, bp, model_subset=model_subset)
    G = X.conj().T @ X
    B = X.conj().T @ Y
    del X
    W = _solve_ridge(G, B, float(ridge_lambda))
    return W, G, B


def fit_ridge_weights_region(
    tx_n,
    rx_residual,
    helpers,
    lags,
    fs_hz,
    train_slice,
    ridge_lambda=1e-6,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
    cross_kinds=fu.DEFAULT_CROSS_KINDS,
    cross_lag_deltas=fu.DEFAULT_CROSS_LAG_DELTAS,
):
    """Same as fit_ridge_weights but train_slice replaces MODEL_SUBSET."""
    return fit_ridge_weights(
        tx_n,
        rx_residual,
        helpers,
        lags,
        fs_hz,
        ridge_lambda=ridge_lambda,
        orders_mem=orders_mem,
        orders_conj=orders_conj,
        use_conjugate=use_conjugate,
        use_cross=use_cross,
        cross_kinds=cross_kinds,
        cross_lag_deltas=cross_lag_deltas,
        model_subset=train_slice,
    )


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
    cross_kinds=fu.DEFAULT_CROSS_KINDS,
    cross_lag_deltas=fu.DEFAULT_CROSS_LAG_DELTAS,
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
        cross_kinds=cross_kinds,
        cross_lag_deltas=cross_lag_deltas,
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
        cross_kinds,
        cross_lag_deltas,
    )
    return rx_res - poly_pred


def sparse_ridge_canceller(
    tx_n,
    rx,
    helpers,
    lags,
    fs_hz,
    ridge_lambda=1e-6,
    top_k=500,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
    cross_kinds=fu.DEFAULT_CROSS_KINDS,
    cross_lag_deltas=fu.DEFAULT_CROSS_LAG_DELTAS,
):
    """Ridge on top-K features by summed band correlation on MODEL_SUBSET."""
    fit_tx = helpers["fit_tx_prediction"]
    bp = make_bp_callable(helpers, fs_hz)
    tx_pred = fit_tx(rx)
    rx_res = rx - tx_pred

    X, F = _build_subset_matrix(
        tx_n,
        bp,
        lags,
        orders_mem=orders_mem,
        orders_conj=orders_conj,
        use_conjugate=use_conjugate,
        use_cross=use_cross,
        cross_kinds=cross_kinds,
        cross_lag_deltas=cross_lag_deltas,
    )
    Y = _stack_targets(rx_res, bp)
    score = np.zeros(F, dtype=np.float64)
    for c in range(4):
        score += np.abs(X.conj().T @ Y[:, c])
    kk = min(int(top_k), F)
    idx = np.argsort(-score)[:kk]
    Xs = X[:, idx]
    del X
    Gs = Xs.conj().T @ Xs
    Bs = Xs.conj().T @ Y
    del Xs
    Ws = _solve_ridge(Gs, Bs, float(ridge_lambda))
    W = np.zeros((F, 4), dtype=np.complex128)
    W[idx, :] = Ws

    poly_pred = predict_poly_from_weights(
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
    cross_kinds=fu.DEFAULT_CROSS_KINDS,
    cross_lag_deltas=fu.DEFAULT_CROSS_LAG_DELTAS,
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
        cross_kinds=cross_kinds,
        cross_lag_deltas=cross_lag_deltas,
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
        cross_kinds=cross_kinds,
        cross_lag_deltas=cross_lag_deltas,
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


def feature_count(
    tx_n,
    lags,
    orders_mem,
    orders_conj,
    use_conjugate,
    use_cross,
    cross_kinds=fu.DEFAULT_CROSS_KINDS,
    cross_lag_deltas=fu.DEFAULT_CROSS_LAG_DELTAS,
):
    return fu.count_feature_columns(
        tx_n.shape[1],
        lags,
        orders_mem,
        orders_conj,
        use_conjugate,
        use_cross,
        cross_kinds=cross_kinds,
        cross_lag_deltas=cross_lag_deltas,
    )
