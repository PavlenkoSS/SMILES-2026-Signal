"""TX-derived feature builders for SIC (lags, memory polynomial, conjugate, cross)."""
import numpy as np

# Cross-channel feature kind names (used in CLI and iterators).
CROSS_KIND_IJ_MAG2 = "ij_mag2"
CROSS_KIND_IJ_MAG4 = "ij_mag4"
CROSS_KIND_CONJ_IJ_MAG2 = "conj_ij_mag2"
CROSS_KIND_CONJ_IJ_MAG4 = "conj_ij_mag4"
DEFAULT_CROSS_KINDS = (CROSS_KIND_IJ_MAG2,)
DEFAULT_CROSS_LAG_DELTAS = (0,)


def shift_signal(x, k):
    y = np.zeros_like(x)
    if k >= 0:
        y[k:] = x[: len(x) - k]
    else:
        kk = -k
        y[: len(x) - kk] = x[kk:]
    return y


def build_lagged_features(tx_n, lags):
    """Per TX channel, shift by each lag. Returns list of (N,) complex arrays."""
    n_tx = tx_n.shape[1]
    feats = []
    for c in range(n_tx):
        for lag in lags:
            feats.append(shift_signal(tx_n[:, c], lag))
    return feats


def build_memory_polynomial_features(tx_n, lags, orders=(1, 3, 5)):
    """Iterator over x[n-k], x[n-k]*|x[n-k]|^2, ... (memory-efficient)."""
    return iter_memory_polynomial_features(tx_n, lags, orders=orders)


def iter_memory_polynomial_features(tx_n, lags, orders=(1, 3, 5)):
    n_tx = tx_n.shape[1]
    for c in range(n_tx):
        for lag in lags:
            z = shift_signal(tx_n[:, c], lag)
            a2 = np.abs(z) ** 2
            for order in orders:
                if order == 1:
                    yield z
                elif order == 3:
                    yield z * a2
                elif order == 5:
                    yield z * (a2**2)
                else:
                    raise ValueError(f"Unsupported order {order}")


def build_conjugate_features(tx_n, lags, orders=(1, 3)):
    """Iterator over conj-based terms."""
    return iter_conjugate_features(tx_n, lags, orders=orders)


def iter_conjugate_features(tx_n, lags, orders=(1, 3, 5)):
    n_tx = tx_n.shape[1]
    for c in range(n_tx):
        for lag in lags:
            z = shift_signal(tx_n[:, c], lag)
            zc = np.conj(z)
            a2 = np.abs(z) ** 2
            for order in orders:
                if order == 1:
                    yield zc
                elif order == 3:
                    yield zc * a2
                elif order == 5:
                    yield zc * (a2**2)
                else:
                    raise ValueError(f"Unsupported conj order {order}")


def _cross_term(zi, zj, kind: str):
    mag2 = np.abs(zj) ** 2
    mag4 = mag2**2
    if kind == CROSS_KIND_IJ_MAG2:
        return zi * mag2
    if kind == CROSS_KIND_IJ_MAG4:
        return zi * mag4
    if kind == CROSS_KIND_CONJ_IJ_MAG2:
        return np.conj(zi) * mag2
    if kind == CROSS_KIND_CONJ_IJ_MAG4:
        return np.conj(zi) * mag4
    raise ValueError(f"Unknown cross kind {kind!r}")


def iter_cross_channel_features(tx_n, lags, kinds=None, lag_deltas=None):
    """Cross-channel terms: z_i at lag k, z_j at lag k+delta (per plan)."""
    if kinds is None:
        kinds = DEFAULT_CROSS_KINDS
    if lag_deltas is None:
        lag_deltas = DEFAULT_CROSS_LAG_DELTAS
    n_tx = tx_n.shape[1]
    for i in range(n_tx):
        for j in range(n_tx):
            if i == j:
                continue
            for lag in lags:
                zi = shift_signal(tx_n[:, i], lag)
                for delta in lag_deltas:
                    zj = shift_signal(tx_n[:, j], lag + int(delta))
                    for kind in kinds:
                        yield _cross_term(zi, zj, kind)


def count_cross_columns(n_tx, lags, kinds, lag_deltas):
    if not kinds:
        return 0
    return n_tx * (n_tx - 1) * len(lags) * len(lag_deltas) * len(kinds)


def count_feature_columns(
    n_tx,
    lags,
    orders_mem,
    orders_conj,
    use_conjugate,
    use_cross,
    cross_kinds=DEFAULT_CROSS_KINDS,
    cross_lag_deltas=DEFAULT_CROSS_LAG_DELTAS,
):
    F = n_tx * len(lags) * len(orders_mem)
    if use_conjugate:
        F += n_tx * len(lags) * len(orders_conj)
    if use_cross:
        F += count_cross_columns(n_tx, lags, cross_kinds, cross_lag_deltas)
    return F


def iter_all_raw_features(
    tx_n,
    lags,
    orders_mem,
    orders_conj,
    use_conjugate,
    use_cross,
    cross_kinds=DEFAULT_CROSS_KINDS,
    cross_lag_deltas=DEFAULT_CROSS_LAG_DELTAS,
):
    yield from iter_memory_polynomial_features(tx_n, lags, orders=orders_mem)
    if use_conjugate:
        yield from iter_conjugate_features(tx_n, lags, orders=orders_conj)
    if use_cross:
        yield from iter_cross_channel_features(
            tx_n, lags, kinds=cross_kinds, lag_deltas=cross_lag_deltas
        )


def bandpass_stack(feat_list, score_filter):
    """Apply scorer bandpass to each raw feature; returns list of (N,) arrays."""
    return [score_filter(f.astype(np.complex128, copy=False)) for f in feat_list]


def subset_matrix_from_filtered(feat_list_filtered, model_subset):
    """Stack columns (n_sub, F) from full-length filtered features."""
    sl = model_subset
    cols = [f[sl].astype(np.complex128, copy=False) for f in feat_list_filtered]
    if not cols:
        return np.zeros((0, 0), dtype=np.complex128)
    return np.column_stack(cols)


def correlate_features_with_target(
    feat_list_filtered, target_band, model_subset, top_k=None
):
    """Rank features by |sum conj(phi)*y| / (||phi||*||y||) on subset (proxy)."""
    sl = model_subset
    y = target_band[sl].astype(np.complex128, copy=False)
    y_norm = np.linalg.norm(y) + 1e-30
    scores = []
    for idx, phi in enumerate(feat_list_filtered):
        p = phi[sl]
        num = np.abs(np.vdot(p, y))
        den = np.linalg.norm(p) * y_norm + 1e-30
        scores.append((num / den, idx))
    scores.sort(key=lambda t: -t[0])
    if top_k is None or top_k >= len(scores):
        order = [i for _, i in scores]
    else:
        order = [i for _, i in scores[:top_k]]
    return order, scores


def select_top_k_columns(X, y_vec, k):
    """X: (n_sub, F), y_vec: (n_sub,) complex — return indices of top-k correlations."""
    if k is None or k >= X.shape[1]:
        return np.arange(X.shape[1])
    norms = np.linalg.norm(X, axis=0) + 1e-30
    yn = np.linalg.norm(y_vec) + 1e-30
    dots = np.abs(X.conj().T @ y_vec) / (norms * yn)
    return np.argsort(-dots)[:k]
