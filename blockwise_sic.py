"""Blockwise overlap-add SIC placeholder: delegates to global ridge for stability."""

from ridge_sic import ridge_canceller


def blockwise_canceller(tx_n, rx, helpers, fs_hz, lags, ridge_lambda=1e-5, **kwargs):
    """Full-sequence ridge; replace with OLA + per-block fits if needed."""
    return ridge_canceller(
        tx_n, rx, helpers, lags, fs_hz, ridge_lambda=ridge_lambda, **kwargs
    )
