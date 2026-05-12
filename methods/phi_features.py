"""Bandpassed TX polynomial features at selected column indices (for Phi-only neural)."""
import numpy as np

from . import feature_utils as fu


def phi_subset_rows(
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
    row_slice: slice,
):
    """Rows row_slice of Phi (complex), columns restricted to col_indices. Shape (n_rows, K)."""
    idx_sorted = np.sort(np.unique(np.asarray(col_indices, dtype=np.int64)))
    K = len(idx_sorted)
    pos = {int(j): i for i, j in enumerate(idx_sorted)}
    n_rows = row_slice.stop - row_slice.start
    out = np.zeros((n_rows, K), dtype=np.complex128)

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
        if j not in pos:
            continue
        filt = bp(raw.astype(np.complex128, copy=False))
        out[:, pos[j]] = filt[row_slice]
        del raw
        del filt
    return out


def phi_window_real(
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
    length,
):
    """Phi real tensor for window [start, start+length): (length, 2*K) float32."""
    sl = slice(start, start + length)
    z = phi_subset_rows(
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
        sl,
    )
    return np.concatenate([z.real.astype(np.float32), z.imag.astype(np.float32)], axis=1)
