"""Fast bandpass matching task_and_baseline (Blackman, 2047 taps, same center/BW)."""
import numpy as np
from scipy.signal import firwin, oaconvolve

from task_and_baseline import BW, CENTER


def scoring_bandpass_kernel(fs_hz, n_taps=2047):
    lp = firwin(n_taps, BW / 2, window="blackman", fs=fs_hz)
    return (lp * np.exp(2j * np.pi * CENTER / fs_hz * np.arange(n_taps))).astype(
        np.complex128
    )


def fast_score_filter(x, kernel):
    x = np.ascontiguousarray(x, dtype=np.complex128)
    return oaconvolve(x, kernel, mode="same")


def make_bp_callable(helpers, fs_hz):
    """Use FFT overlap-add for feature/target filtering; falls back if fs_hz is None."""
    if fs_hz is None:
        return helpers["score_filter"]
    ker = scoring_bandpass_kernel(fs_hz)
    return lambda z: fast_score_filter(z, ker)
