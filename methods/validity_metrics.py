import numpy as np

from .rank1_sic import rank1_from_band_matrix
from task_and_baseline import MAX_UNEXPLAINED_TO_RESIDUAL, MIN_EXPLAIN_RATIO


def validity_report(helpers, rx_before, rx_after):
    score_filter = helpers["score_filter"]
    fit_tx = helpers["fit_tx_prediction"]

    removed_band = np.column_stack(
        [score_filter(rx_before[:, ch] - rx_after[:, ch]) for ch in range(rx_before.shape[1])]
    )
    tx_part = fit_tx(rx_before - rx_after)
    residual = removed_band - tx_part
    rank1_part = rank1_from_band_matrix(residual)
    err = residual - rank1_part

    total_power = np.mean(np.abs(removed_band) ** 2) + 1e-30
    err_power = np.mean(np.abs(err) ** 2)
    explain_ratio = float(1.0 - err_power / total_power)

    residual_band = np.column_stack(
        [score_filter(rx_after[:, ch]) for ch in range(rx_after.shape[1])]
    )
    err_powers = np.mean(np.abs(err) ** 2, axis=0)
    residual_powers = np.mean(np.abs(residual_band) ** 2, axis=0) + 1e-30
    residual_guard = bool(
        np.all(err_powers <= MAX_UNEXPLAINED_TO_RESIDUAL * residual_powers)
    )
    valid = explain_ratio >= MIN_EXPLAIN_RATIO and residual_guard

    reds = []
    for ch in range(4):
        p0 = np.mean(np.abs(score_filter(rx_before[:, ch])) ** 2)
        p1 = np.mean(np.abs(score_filter(rx_after[:, ch])) ** 2) + 1e-30
        r = 10 * np.log10(p0 / p1) if valid else 0.0
        reds.append(float(r))
    avg = float(np.mean(reds))
    return {
        "explain_ratio": explain_ratio,
        "valid": valid,
        "per_channel_db": reds,
        "average_db": avg,
        "residual_guard": residual_guard,
    }
