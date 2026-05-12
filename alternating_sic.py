"""Alternate ridge refit on cleaned RX and rank-1 subtraction."""
from rank1_sic import grid_search_rank1_alpha
from ridge_sic import fit_ridge_weights, predict_poly_from_weights


def alternating_canceller(
    tx_n,
    rx,
    helpers,
    fs_hz,
    n_iter=3,
    lags=None,
    ridge_lambda=1e-5,
    rank1_alphas=None,
    orders_mem=(1, 3, 5),
    orders_conj=(1, 3),
    use_conjugate=True,
    use_cross=False,
):
    if lags is None:
        lags = list(range(-16, 17))
    from sic_filters import make_bp_callable

    fit_tx = helpers["fit_tx_prediction"]
    bp = make_bp_callable(helpers, fs_hz)
    tx_pred = fit_tx(rx)
    rx_curr = rx - tx_pred

    for _ in range(n_iter):
        W, _, _ = fit_ridge_weights(
            tx_n,
            rx_curr,
            helpers,
            lags,
            fs_hz,
            ridge_lambda=ridge_lambda,
            orders_mem=orders_mem,
            orders_conj=orders_conj,
            use_conjugate=use_conjugate,
            use_cross=use_cross,
        )
        poly = predict_poly_from_weights(
            tx_n,
            bp,
            lags,
            W,
            orders_mem,
            orders_conj,
            use_conjugate,
            use_cross,
        )
        rx_curr = rx_curr - poly

        rx_curr, _, _ = grid_search_rank1_alpha(
            rx, rx_curr, helpers, alphas=rank1_alphas
        )

    return rx_curr
